import logging
import time
from pathlib import Path

from easywhisper.pipelines import pipeline
from easywhisper.text.normalization import text_normalizer

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    AUDIO_DIR = "data/multi"
    audio_files = [file.name for file in Path(AUDIO_DIR).glob("*.wav")]
    start = time.time()
    pipeline(
        vad_model="pyannote",
        emissions_model="facebook/mms-1b-all",
        transcription_model="openai/whisper-large-v3",
        audio_paths=audio_files,
        audio_dir=AUDIO_DIR,
        language=None,
        batch_size_features=16,
        text_normalizer_fn=text_normalizer,
    )
    end = time.time()
    logger.info(f"Total time: {end - start}")
