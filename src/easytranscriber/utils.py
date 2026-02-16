import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def hf_to_ct2_converter(model_path, cache_dir="models"):
    model_name = Path(model_path).stem
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
