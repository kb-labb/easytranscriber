import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def hf_to_ct2_converter(model_path, cache_dir="models"):
    """
    Convert a Hugging Face Transformers model to CTranslate2 format.

    Saves the converted model to the specified cache directory.
    If the converted model already exists, it will be reused.

    Parameters
    ----------
    model_path : str
        The Hugging Face model identifier or local path to the model to be converted.
    cache_dir : str, optional
        The directory where the converted CTranslate2 model will be saved. Default is "models".

    Returns
    -------
    str
        The path to the converted CTranslate2 model directory.

    Example
    -------
    ```python
    ct2_model_path = hf_to_ct2_converter("KBLab/kb-whisper-large")
    ```
    """
    p = Path(model_path)
    # Handle model names like "distil-whisper/distil-large-v3.5" with a false suffix
    model_name = p.stem + p.suffix if p.suffix and p.suffix[1:].isdigit() else p.stem
    output_dir = Path(cache_dir) / "ct2" / model_name

    if output_dir.exists():
        logger.info(f"Using existing ctranslate2 model from {output_dir}")
        return output_dir

    try:
        logger.info(
            (
                f"Downloading {model_path} from HF and converting to ctranslate2 model...\n"
                f"Saving to {output_dir}\n"
            )
        )
        subprocess.run(
            [
                "ct2-transformers-converter",
                "--model",
                model_path,
                "--output_dir",
                str(output_dir),
                "--copy_files",
                "tokenizer.json",
                "--copy_files",
                "preprocessor_config.json",
                "--quantization",
                "float16",
            ],
            check=True,
        )
        logger.info(f"CT2 model saved to {output_dir}")
        return output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during conversion: {e.stderr.decode()}")
        exit(1)
