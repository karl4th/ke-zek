"""Domain augmentation for K-T1: one "channel" per recording.

A ``Chain`` is drawn once per window and applied to the whole conversation:

* ``Chain.sources`` is the linear part (room reverb, telephone band). It is applied to each
  speaker's track separately, which equals applying it to their sum, so the clean target for the
  loss carries the same reverb and band as the mixture.
* ``Chain.finish`` is the rest, applied to the summed mixture only: background noise and music
  (through the same band), gain, clipping, a lossy codec. The clean target never gets these.

20% of windows are clean (no augmentation at all). MUSAN speech is never used as background:
a third talker would teach the model to keep other voices.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

SR = 16000


@dataclass(frozen=True)
class AugmentConfig:
    clean_share: float = 0.2
    reverb_p: float = 0.4
    phone_p: float = 0.2
    noise_p: float = 0.5
    noise_snr_db: tuple[float, float] = (5.0, 30.0)
    music_p: float = 0.2
    music_rel_db: tuple[float, float] = (-30.0, -15.0)
    gain_db: float = 6.0
    clip_p: float = 0.1
    clip_over_db: tuple[float, float] = (1.0, 6.0)
    codec_p: float = 0.3
    mp3_kbps: tuple[int, int] = (48, 128)
    opus_kbps: tuple[int, int] = (16, 32)


AUDIO_EXT = (".wav", ".flac")


def _is_speech(path: str) -> bool:
    return "speech" in os.path.normpath(path).lower().split(os.sep)


def _files(dirs: list[str]) -> list[str]:
    out = []
    for d in dirs:
        out += [
            f
            for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
            if f.lower().endswith(AUDIO_EXT)
        ]
    return sorted(out)


RIR_CACHE = 2000  # RIRs kept in memory per bank


@dataclass
class DomainBank:
    rirs: list[str]
    noises: list[str]
    music: list[str]

    @classmethod
    def from_dirs(cls, rir_dirs=(), noise_dirs=(), music_dirs=()) -> DomainBank:
        noises, music = _files(list(noise_dirs)), _files(list(music_dirs))
        bad = [f for f in noises + music if _is_speech(f)]
        if bad:
            raise ValueError(f"speech files in the noise/music bank, e.g. {bad[0]}")
        # RIRS_NOISES/real_rirs_isotropic_noises keeps room noise recordings next to the RIRs
        rirs = [f for f in _files(list(rir_dirs)) if "noise" not in os.path.basename(f).lower()]
        return cls(rirs=rirs, noises=noises, music=music)

    def __post_init__(self):
        self._rir_cache: dict[str, np.ndarray] = {}

    def rir(self, path: str) -> np.ndarray:
        # bounded: the simulated RIR set alone is several GB per DataLoader worker as float64
        if path not in self._rir_cache:
            if len(self._rir_cache) >= RIR_CACHE:
                self._rir_cache.pop(next(iter(self._rir_cache)))
            self._rir_cache[path] = _read_mono(path).astype(np.float32)
        return self._rir_cache[path].astype(np.float64)


def _read_mono(path: str, start: int = 0, frames: int = -1) -> np.ndarray:
    x, sr = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    x = x[:, 0].astype(np.float64)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    return x


def read_segment(path: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """A random ``n``-sample stretch of a file at 16 kHz, looped when the file is shorter."""
    info = sf.info(path)
    n_src = int(np.ceil(n * info.samplerate / SR)) + 1
    if info.frames > n_src:
        s = int(rng.integers(0, info.frames - n_src + 1))
        x = _read_mono(path, s, n_src)
    else:
        x = _read_mono(path)
        x = np.tile(x, int(np.ceil(n / max(len(x), 1))) + 1)
        x = np.roll(x, -int(rng.integers(0, max(len(x) // 2, 1))))
    return x[:n]


_PHONE = butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos")


def phone(x: np.ndarray) -> np.ndarray:
    y = sosfilt(_PHONE, x)
    return resample_poly(resample_poly(y, 1, 2), 2, 1)[: len(x)]


def reverb(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve and keep the direct path aligned with the dry signal."""
    d = int(np.argmax(np.abs(rir)))
    rir = rir / (np.abs(rir[d]) + 1e-12)
    return fftconvolve(x, rir)[d : d + len(x)]


def codec(x: np.ndarray, kind: str, kbps: int) -> np.ndarray:
    """Encode and decode with ffmpeg (libmp3lame / libopus); the result is re-aligned to ``x``."""
    if kind == "mp3":
        enc = ["-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-f", "mp3"]
    elif kind == "opus":
        enc = ["-c:a", "libopus", "-b:a", f"{kbps}k", "-f", "ogg"]
    else:
        raise ValueError(kind)
    raw = ["-f", "f32le", "-ar", str(SR), "-ac", "1"]
    coded = subprocess.run(
        ["ffmpeg", "-v", "error", *raw, "-i", "pipe:0", *enc, "pipe:1"],
        input=np.float32(x).tobytes(),
        capture_output=True,
        check=True,
    ).stdout
    y = np.frombuffer(
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", *raw, "pipe:1"],
            input=coded,
            capture_output=True,
            check=True,
        ).stdout,
        dtype=np.float32,
    ).astype(np.float64)
    max_lag = 4096
    y = np.pad(y, (0, max(0, len(x) + max_lag - len(y))))
    seg = x[: min(len(x), 4 * SR)]
    corr = fftconvolve(y[: len(seg) + max_lag], seg[::-1], mode="valid")
    lag = int(np.argmax(corr[: max_lag + 1]))
    return y[lag : lag + len(x)]


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


@dataclass
class Chain:
    clean: bool = True
    rir: np.ndarray | None = None
    phone: bool = False
    noise: str | None = None
    snr_db: float = 0.0
    music: str | None = None
    music_db: float = 0.0
    gain_db: float = 0.0
    clip_over_db: float | None = None
    codec: tuple[str, int] | None = None

    def sources(self, x: np.ndarray) -> np.ndarray:
        """Linear part, per speaker track."""
        y = np.asarray(x, dtype=np.float64)
        if self.rir is not None:
            y = reverb(y, self.rir)
        if self.phone:
            y = phone(y)
        return y

    def finish(
        self, mix: np.ndarray, speech_energy: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, float]:
        """Background, gain, clipping, codec on the summed mixture. Returns ``(mixture, gain)``:
        the linear gain the speech got, to apply to the clean references."""
        out = np.asarray(mix, dtype=np.float64).copy()
        for path, rel_db in ((self.noise, -self.snr_db), (self.music, self.music_db)):
            if path is None:
                continue
            b = read_segment(path, len(out), rng)
            if self.phone:
                b = phone(b)
            b *= np.sqrt(speech_energy * 10 ** (rel_db / 10) / (np.mean(b**2) + 1e-12))
            out += b
        g = 10 ** (self.gain_db / 20)
        out *= g
        if self.clip_over_db is not None:
            k = 10 ** (self.clip_over_db / 20) / max(np.abs(out).max(), 1e-9)
            out, g = np.clip(out * k, -1.0, 1.0), g * k
        if self.codec is not None:
            out = codec(out, *self.codec)
        return out, g


def sample_chain(
    bank: DomainBank | None, rng: np.random.Generator, cfg: AugmentConfig | None = None
) -> Chain:
    """One channel for one window. ``bank=None`` or an empty bank only allows the clean chain
    and the parts that need no files (band, gain, clipping, codec)."""
    cfg = cfg or AugmentConfig()
    if rng.random() < cfg.clean_share:
        return Chain(clean=True)
    c = Chain(clean=False)
    if bank and bank.rirs and rng.random() < cfg.reverb_p:
        c.rir = bank.rir(bank.rirs[int(rng.integers(len(bank.rirs)))])
    c.phone = bool(rng.random() < cfg.phone_p)
    if bank and bank.noises and rng.random() < cfg.noise_p:
        c.noise = bank.noises[int(rng.integers(len(bank.noises)))]
        c.snr_db = float(rng.uniform(*cfg.noise_snr_db))
    if bank and bank.music and rng.random() < cfg.music_p:
        c.music = bank.music[int(rng.integers(len(bank.music)))]
        c.music_db = float(rng.uniform(*cfg.music_rel_db))
    c.gain_db = float(rng.uniform(-cfg.gain_db, cfg.gain_db))
    if rng.random() < cfg.clip_p:
        c.clip_over_db = float(rng.uniform(*cfg.clip_over_db))
    if rng.random() < cfg.codec_p:
        if rng.random() < 0.5:
            c.codec = ("mp3", int(rng.integers(cfg.mp3_kbps[0], cfg.mp3_kbps[1] + 1)))
        else:
            c.codec = ("opus", int(rng.integers(cfg.opus_kbps[0], cfg.opus_kbps[1] + 1)))
    return c
