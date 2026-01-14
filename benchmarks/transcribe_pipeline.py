import logging
import time
from pathlib import Path

from easywhisper.pipelines import pipeline

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    AUDIO_DIR = "data/sv"
    audio_files = [file.name for file in Path(AUDIO_DIR).glob("*")]
    start = time.time()
    pipeline(
        vad_model="pyannote",
        emissions_model="KBLab/wav2vec2-large-voxrex-swedish",
        transcription_model="KBLab/kb-whisper-large",
        audio_paths=audio_files,
        audio_dir=AUDIO_DIR,
        language="sv",
        batch_size_features=16,
    )
    end = time.time()
    logger.info(f"Total time: {end - start}")
