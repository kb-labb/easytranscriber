import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from easytranscriber.utils import hf_to_ct2_converter


@patch("easytranscriber.utils.subprocess.run")
@patch("easytranscriber.utils.Path.exists")
def test_hf_to_ct2_converter_existing(mock_exists, mock_run):
    # If output directory already exists, return early
    mock_exists.return_value = True

    output_dir = hf_to_ct2_converter("org/my-model", cache_dir="my_cache")

    # Should not invoke subprocess
    mock_run.assert_not_called()
    assert str(output_dir).endswith("my-model")


@patch("easytranscriber.utils.subprocess.run")
@patch("easytranscriber.utils.Path.exists")
def test_hf_to_ct2_converter_new(mock_exists, mock_run):
    # If output directory does not exist, run conversion
    mock_exists.return_value = False

    output_dir = hf_to_ct2_converter("org/my-model", cache_dir="my_cache")

    # Should invoke subprocess
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]

    assert cmd[0] == "ct2-transformers-converter"
    assert "--model" in cmd
    assert "org/my-model" in cmd
    assert "--output_dir" in cmd
    assert "--quantization" in cmd
    assert "float16" in cmd
    assert str(output_dir).endswith("my-model")


@patch("easytranscriber.utils.subprocess.run")
@patch("easytranscriber.utils.Path.exists")
def test_hf_to_ct2_converter_failure(mock_exists, mock_run):
    mock_exists.return_value = False

    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="ct2-transformers-converter", stderr=b"Conversion failed"
    )

    with pytest.raises(SystemExit):
        hf_to_ct2_converter("org/my-model", cache_dir="my_cache")
