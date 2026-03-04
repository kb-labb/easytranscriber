import logging
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def read_audio_segment(
    audio_path: str | Path,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Read a segment of audio using ffmpeg subprocess with seek.

    Uses ffmpeg's fast seek (-ss before -i) to efficiently read only the
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
        Target sample rate for resampling.

    Returns
    -------
    np.ndarray
        Audio data as float32 numpy array.
    """
    cmd = [
        "ffmpeg",
        "-ss",
        str(start_sec),  # Seek to position (before -i = fast seek)
        "-i",
        str(audio_path),
        "-t",
        str(duration_sec),  # Read this many seconds
        "-ar",
        str(sample_rate),  # Resample
        "-ac",
        "1",  # Mono
        "-f",
        "f32le",  # Raw float32 little-endian output
        "-loglevel",
        "error",
        "pipe:1",  # Output to stdout
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        return audio
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg error reading {audio_path}: {e.stderr.decode()}")
        raise


def convert_audio_to_array(input_file: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Convert audio to in-memory numpy array.

    Parameters
    ----------
    input_file : str
        Path to the input audio file.
    sample_rate : int, optional
        Target sample rate.

    Returns
    -------
    Tuple[np.ndarray, int]
        Tuple containing the audio array (int16) and the sample rate.

    Raises
    ------
    RuntimeError
        If ffmpeg command fails.
    """
    # fmt: off
    command = [
        "ffmpeg",
        "-i", input_file,
        "-f", "s16le",  # raw PCM 16-bit little endian
        "-acodec", "pcm_s16le",
        "-ac", "1",  # mono
        "-ar", str(sample_rate),  # 16 kHz
        "-loglevel", "error",  # suppress output
        "-hide_banner",
        "-nostats",
    ]
    # fmt: on

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {err.decode()}")

    # Convert byte output to numpy array
    audio_array = np.frombuffer(out, dtype=np.int16)

    return audio_array, sample_rate  # (samples, sample_rate)


def convert_audio_to_wav(input_file: str, output_file: str) -> None:
    """
    Convert audio file to WAV format with 16kHz sample rate and mono channel.

    Parameters
    ----------
    input_file : str
        Path to the input audio file.
    output_file : str
        Path to the output WAV file.

    Raises
    ------
    RuntimeError
        If ffmpeg command fails.
    """
    # fmt: off
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ar', '16000',  # Set the audio sample rate to 16kHz
        '-ac', '1',      # Set the number of audio channels to 1 (mono)
        '-c:a', 'pcm_s16le',
        '-loglevel', 'warning',
        '-hide_banner',
        '-nostats',
        '-nostdin',
        output_file
    ]
    # fmt: on
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {err.decode()}")
