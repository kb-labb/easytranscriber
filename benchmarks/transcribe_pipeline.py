import logging
import time
from pathlib import Path

from easyaligner.text import load_tokenizer

from easytranscriber.pipelines import pipeline
from easytranscriber.text.normalization import text_normalizer

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

tokenizer = load_tokenizer("english")
if __name__ == "__main__":
    AUDIO_DIR = "data/en"
    audio_files = [file.name for file in Path(AUDIO_DIR).glob("*.wav")]
    start = time.time()
    pipeline(
        vad_model="pyannote",
        emissions_model="facebook/wav2vec2-base-960h",
        transcription_model="distil-whisper/distil-large-v3.5",
        audio_paths=audio_files,
        audio_dir=AUDIO_DIR,
        backend="ct2",
        language="en",
        tokenizer=tokenizer,
        batch_size_features=16,
        text_normalizer_fn=text_normalizer,
        save_emissions=True,
        delete_emissions=True,
        device="cpu",
    )
    end = time.time()
    logger.info(f"Total time: {end - start}")
