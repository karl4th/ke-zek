"""Extract one speaker from a mono recording with a trained v2 checkpoint."""

import argparse

from kezek.tse_v2_infer import extract_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--enrollment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    extract_file(args.checkpoint, args.mixture, args.enrollment, args.output, args.device)


if __name__ == "__main__":
    main()
