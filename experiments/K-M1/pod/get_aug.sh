#!/usr/bin/env bash
# MUSAN (noise, music; speech is never used) and RIRS_NOISES for the recording render chain.
set -e
mkdir -p /data/aug/dl && cd /data/aug
O=https://openslr.elda.org/resources
[ -d RIRS_NOISES ] || { aria2c -q -x16 -s16 -d dl $O/28/rirs_noises.zip && unzip -q -o dl/rirs_noises.zip -d . && rm dl/rirs_noises.zip; }
[ -d musan ] || { aria2c -q -x16 -s16 -d dl $O/17/musan.tar.gz && pigz -dc dl/musan.tar.gz | tar -x -C . && rm dl/musan.tar.gz; }
du -sh /data/aug/*
echo AUG READY
