import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def read_audio_segment(
    audio_path: str | Path,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Read a segment of audio using torchaudio with seek.

    Uses `torchaudio.load` with `frame_offset` and `num_frames` to efficiently read only the
    required segment, with resampling to the target sample rate and mono conversion.

    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file.
    start_sec : float
        Start time in seconds.
    duration_sec : float
        Duration to read in seconds.
    sample_rate : int, optional
        Target sample rate for resampling. Default is 16000.

    Returns
    -------
    np.ndarray
        Audio data as float32 numpy array.
    """
    audio_str_path = str(audio_path)

    # Get audio metadata to calculate frames
    info = sf.info(audio_str_path)
    original_sr = info.samplerate

    frame_offset = int(start_sec * original_sr)
    num_frames = int(duration_sec * original_sr)

    # Load only the specific frames
    audio, sr = torchaudio.load(
        audio_str_path,
        frame_offset=frame_offset,
        num_frames=num_frames,
    )

    # Convert to mono if necessary
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        audio = resampler(audio)

    # Output should be 1D numpy array of type float32 (torchaudio naturally loads as float32)
    return audio.squeeze(0).numpy().astype(np.float32)


def convert_audio_to_array(input_file: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Convert audio to in-memory int16 numpy array using torchaudio.

    Parameters
    ----------
    input_file : str
        Path to the input audio file.
    sample_rate : int, optional
        Target sample rate. Default is 16000.

    Returns
    -------
    Tuple[np.ndarray, int]
        Tuple containing the audio array (int16) and the sample rate.
    """
    audio, sr = torchaudio.load(input_file)

    # Convert to mono if necessary
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        audio = resampler(audio)

    # torchaudio loads as float32 in [-1.0, 1.0]. Convert to int16.
    # We clip to avoid wrap-around on exact 1.0/-1.0
    audio = torch.clamp(audio * 32768.0, min=-32768.0, max=32767.0)
    audio_array = audio.squeeze(0).numpy().astype(np.int16)

    return audio_array, sample_rate


def convert_audio_to_wav(input_file: str, output_file: str) -> None:
    """
    Convert audio file to WAV format with 16kHz sample rate and mono channel using torchaudio.

    Parameters
    ----------
    input_file : str
        Path to the input audio file.
    output_file : str
        Path to the output WAV file.
    """
    audio, sr = torchaudio.load(input_file)

    # Convert to mono if necessary
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)
        sr = target_sr

    # Save to wav (PCM 16-bit)
    torchaudio.save(
        output_file, audio, sample_rate=sr, format="wav", encoding="PCM_S", bits_per_sample=16
    )
