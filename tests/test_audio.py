import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from easytranscriber.audio import read_audio_segment, convert_audio_to_array, convert_audio_to_wav


@patch("easytranscriber.audio.subprocess.run")
def test_read_audio_segment_success(mock_run):
    # Mocking stdout to return 10 float32 zeros
    mock_run.return_value.stdout = np.zeros(10, dtype=np.float32).tobytes()
    mock_run.return_value.returncode = 0

    audio = read_audio_segment("dummy.wav", 0.0, 1.0)

    # Check that subprocess.run was called with correct command
    cmd = mock_run.call_args[0][0]
    assert "ffmpeg" == cmd[0]
    assert "dummy.wav" in cmd
    assert "-ss" in cmd

    # Audio should be a numpy array of float32
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == 10


@patch("easytranscriber.audio.subprocess.run")
def test_read_audio_segment_failure(mock_run):
    # Simulate an ffmpeg error
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="ffmpeg", stderr=b"Command failed"
    )

    with pytest.raises(RuntimeError) as excinfo:
        read_audio_segment("dummy.wav", 0.0, 1.0)

    assert "ffmpeg error: Command failed" in str(excinfo.value)


@patch("easytranscriber.audio.subprocess.run")
def test_convert_audio_to_array(mock_run):
    # Mocking stdout to return 10 int16 zeros
    mock_run.return_value.stdout = np.zeros(10, dtype=np.int16).tobytes()
    mock_run.return_value.returncode = 0

    audio_array, sr = convert_audio_to_array("dummy.wav", sample_rate=16000)

    # Verify return types and values
    assert isinstance(audio_array, np.ndarray)
    assert audio_array.dtype == np.int16
    assert len(audio_array) == 10
    assert sr == 16000


@patch("easytranscriber.audio.subprocess.run")
def test_convert_audio_to_wav(mock_run):
    mock_run.return_value.stdout = b""
    mock_run.return_value.returncode = 0

    convert_audio_to_wav("input.mp3", "output.wav")

    cmd = mock_run.call_args[0][0]
    assert "ffmpeg" in cmd
    assert "input.mp3" in cmd
    assert "output.wav" in cmd
    assert "-y" in cmd  # should overwrite output
