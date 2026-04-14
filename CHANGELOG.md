# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added unit tests for `easytranscriber.audio` utilizing `unittest.mock` to mock `torchaudio` and `soundfile` dependencies natively.
- Added unit tests for `easytranscriber.utils` to test `hf_to_ct2_converter` edge cases.

### Changed

- **Major Refactor (`pipelines.py`)**: Split the monolithic `pipeline()` function into discrete, single-responsibility helper operations (`_run_vad`, `_run_transcription`, `_run_emissions`, `_run_alignment`).
- **Torchaudio Migration (`audio.py`)**: Completely abstracted out `ffmpeg` subprocess calls and replaced them natively with `torchaudio` and `soundfile`.
  - `read_audio_segment()` uses `torchaudio.load()` with frame offsets for efficient seeking.
  - `convert_audio_to_array()` loads audio into PyTorch memory and resamples dynamically.
  - `convert_audio_to_wav()` saves output via `torchaudio.save()` without shelling out.

### Notes on Dependencies

- **FFMPEG Exception**: While `easytranscriber` itself is now fully clean of native `ffmpeg` dependencies, the underlying `easyaligner` package internal dataloaders still require `ffmpeg` to exist on the OS system path for complete execution of the forced-alignment stages.
