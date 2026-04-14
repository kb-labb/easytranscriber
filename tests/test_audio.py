import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from easytranscriber.audio import read_audio_segment, convert_audio_to_array, convert_audio_to_wav


@pytest.fixture
def mock_soundfile_info():
    with patch("easytranscriber.audio.sf.info") as mock_info:
        info_mock = MagicMock()
        info_mock.samplerate = 16000
        mock_info.return_value = info_mock
        yield mock_info


@pytest.fixture
def mock_torchaudio_load():
    with patch("easytranscriber.audio.torchaudio.load") as mock_load:
        # Mock returning a 1-second 16kHz mono audio tensor (float32) and sample rate
        audio_tensor = torch.zeros((1, 16000), dtype=torch.float32)
        mock_load.return_value = (audio_tensor, 16000)
        yield mock_load


@pytest.fixture
def mock_torchaudio_transform():
    with patch("easytranscriber.audio.torchaudio.transforms.Resample") as mock_resample:
        mock_resample.return_value = lambda x: x  # Identity mock
        yield mock_resample


@pytest.fixture
def mock_torchaudio_save():
    with patch("easytranscriber.audio.torchaudio.save") as mock_save:
        yield mock_save


def test_read_audio_segment_success(mock_soundfile_info, mock_torchaudio_load):
    audio = read_audio_segment("dummy.wav", 0.0, 1.0)

    mock_soundfile_info.assert_called_once_with("dummy.wav")
    mock_torchaudio_load.assert_called_once_with("dummy.wav", frame_offset=0, num_frames=16000)

    # Audio should be a numpy array of float32
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == 16000


def test_convert_audio_to_array(mock_torchaudio_load):
    audio_array, sr = convert_audio_to_array("dummy.wav", sample_rate=16000)

    mock_torchaudio_load.assert_called_once_with("dummy.wav")

    # Verify return types and values
    assert isinstance(audio_array, np.ndarray)
    assert audio_array.dtype == np.int16
    assert len(audio_array) == 16000
    assert sr == 16000


def test_convert_audio_to_array_resample(mock_torchaudio_load, mock_torchaudio_transform):
    # Setup load to return 44.1kHz audio
    audio_tensor = torch.zeros((1, 44100), dtype=torch.float32)
    mock_torchaudio_load.return_value = (audio_tensor, 44100)

    convert_audio_to_array("dummy.wav", sample_rate=16000)

    mock_torchaudio_transform.assert_called_once_with(orig_freq=44100, new_freq=16000)


def test_convert_audio_to_wav(mock_torchaudio_load, mock_torchaudio_save):
    convert_audio_to_wav("input.mp3", "output.wav")

    mock_torchaudio_load.assert_called_once_with("input.mp3")
    mock_torchaudio_save.assert_called_once()

    args, kwargs = mock_torchaudio_save.call_args
    assert args[0] == "output.wav"
    assert kwargs["sample_rate"] == 16000
    assert kwargs["format"] == "wav"
