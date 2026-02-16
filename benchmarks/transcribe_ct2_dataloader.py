import logging
import time
from pathlib import Path

import ctranslate2
import torch
from easyaligner.data.collators import audiofile_collate_fn, metadata_collate_fn
from easyaligner.data.dataset import AudioFileDataset, JSONMetadataDataset
from easyaligner.pipelines import alignment_pipeline, emissions_pipeline, vad_pipeline
from easyaligner.text.normalization import SpanMapNormalizer
from easyaligner.vad.pyannote import load_vad_model
from easytranscriber.asr.ct2 import transcribe
from easytranscriber.data import StreamingAudioFileDataset
from transformers import AutoModelForCTC, Wav2Vec2Processor, WhisperProcessor

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def text_normalizer(text: str) -> str:
    """
    Normalize text for forced alignment.

    Removes punctuation, normalizes whitespace, and lowercases text
    to prepare it for alignment with wav2vec2.
    """
    normalizer = SpanMapNormalizer(text)
    normalizer.transform(r"\(.*?\)", "")  # Remove parentheses and their content
    normalizer.transform(r"\s[^\w\s]\s", " ")  # Remove punctuation between whitespace
    normalizer.transform(r"[^\w\s]", "")  # Remove punctuation and special characters
    normalizer.transform(r"\s+", " ")  # Normalize whitespace to a single space
    normalizer.transform(r"^\s+|\s+$", "")  # Strip leading and trailing whitespace
    normalizer.transform(r"\w+", lambda m: m.group().lower())

    mapping = normalizer.get_token_map()
    normalized_tokens = [item["normalized_token"] for item in mapping]
    return normalized_tokens, mapping


if __name__ == "__main__":
    # Configuration
    AUDIO_DIR = "data/sv"
    VAD_OUTPUT_DIR = "output/vad"
    TRANSCRIPTION_OUTPUT_DIR = "output/transcriptions"
    EMISSIONS_OUTPUT_DIR = "output/emissions"
    ALIGNMENT_OUTPUT_DIR = "output/alignments"
    MODEL_PATH = "models/kb-whisper-large"
    HF_MODEL_ID = "KBLab/kb-whisper-large"
    WAV2VEC2_MODEL_ID = "KBLab/wav2vec2-large-voxrex-swedish"
    LANGUAGE = "sv"
    USE_STREAMING = True  # Set to False to use AudioFileDataset from easyaligner

    # Audio files to process (can be modified to accept CLI args)
    audio_files = [file.name for file in Path(AUDIO_DIR).glob("*")]

    # =========================================================================
    # Step 1: Run VAD to get speech segments
    # =========================================================================
    logger.info("Loading VAD model...")
    model_vad = load_vad_model()

    # Time with time
    start_time = time.time()

    logger.info("Running VAD pipeline...")
    vad_outputs = vad_pipeline(
        model=model_vad,
        audio_paths=audio_files,
        audio_dir=AUDIO_DIR,
        speeches=None,
        chunk_size=30,
        sample_rate=16000,
        metadata=None,
        batch_size=1,
        num_workers=1,
        prefetch_factor=2,
        save_json=True,
        save_msgpack=False,
        output_dir=VAD_OUTPUT_DIR,
    )

    # =========================================================================
    # Step 2: Load CTranslate2 model and processor
    # =========================================================================
    logger.info(f"Loading CTranslate2 model from {MODEL_PATH}...")
    model = ctranslate2.models.Whisper(MODEL_PATH, device="cuda")

    logger.info(f"Loading processor from {HF_MODEL_ID}...")
    processor = WhisperProcessor.from_pretrained(HF_MODEL_ID)

    # =========================================================================
    # Step 3: Create dataset and dataloader
    # =========================================================================

    json_dataset = JSONMetadataDataset(json_paths=list(Path(VAD_OUTPUT_DIR).rglob("*.json")))

    if USE_STREAMING:
        # Use the streaming dataset that reads chunks on-demand via ffmpeg
        logger.info("Using StreamingAudioFileDataset for memory-efficient loading...")
        file_dataset = StreamingAudioFileDataset(
            metadata=json_dataset,
            processor=processor,
            audio_dir=AUDIO_DIR,
            sample_rate=16000,
            chunk_size=30,
            alignment_strategy="chunk",
        )

        file_dataloader = torch.utils.data.DataLoader(
            file_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=audiofile_collate_fn,
            num_workers=2,
            prefetch_factor=2,
        )
    else:
        # Use the easyaligner AudioFileDataset
        logger.info("Using AudioFileDataset from easyaligner...")
        file_dataset = AudioFileDataset(
            metadata=json_dataset,
            processor=processor,
            sample_rate=16000,
            chunk_size=30,
            alignment_strategy="chunk",
        )

        file_dataloader = torch.utils.data.DataLoader(
            file_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=audiofile_collate_fn,
            num_workers=2,
            prefetch_factor=2,
        )

    # =========================================================================
    # Step 4: Transcribe
    # =========================================================================
    logger.info("Starting transcription...")
    transcribe(
        model=model,
        processor=processor,
        file_dataloader=file_dataloader,
        batch_size=16,
        output_dir=TRANSCRIPTION_OUTPUT_DIR,
        language=LANGUAGE,
        task="transcribe",
        beam_size=5,
    )

    logger.info(f"Transcription complete! Output saved to {TRANSCRIPTION_OUTPUT_DIR}")

    # =========================================================================
    # Step 5: Compute wav2vec2 emissions for alignment
    # =========================================================================
    logger.info(f"Loading wav2vec2 model from {WAV2VEC2_MODEL_ID}...")
    model_w2v = AutoModelForCTC.from_pretrained(WAV2VEC2_MODEL_ID).to("cuda").half()
    processor_w2v = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_ID)

    json_dataset = JSONMetadataDataset(
        json_paths=list(Path(TRANSCRIPTION_OUTPUT_DIR).rglob("*.json"))
    )

    logger.info("Computing emissions...")
    emissions_output = emissions_pipeline(
        model=model_w2v,
        processor=processor_w2v,
        metadata=json_dataset,
        audio_dir=AUDIO_DIR,
        sample_rate=16000,
        chunk_size=30,
        alignment_strategy="chunk",
        batch_size_files=1,
        num_workers_files=2,
        prefetch_factor_files=2,
        batch_size_features=4,
        num_workers_features=4,
        save_json=True,
        save_msgpack=False,
        save_emissions=True,
        return_emissions=True,
        output_dir=EMISSIONS_OUTPUT_DIR,
    )

    logger.info(f"Emissions saved to {EMISSIONS_OUTPUT_DIR}")

    # =========================================================================
    # Step 6: Run forced alignment
    # =========================================================================
    logger.info("Running forced alignment...")
    json_dataset = JSONMetadataDataset(json_paths=list(Path(EMISSIONS_OUTPUT_DIR).rglob("*.json")))
    audiometa_loader = torch.utils.data.DataLoader(
        json_dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=metadata_collate_fn,
    )

    alignments = alignment_pipeline(
        dataloader=audiometa_loader,
        text_normalizer_fn=text_normalizer,
        processor=processor_w2v,
        tokenizer=None,
        emissions_dir=EMISSIONS_OUTPUT_DIR,
        output_dir=ALIGNMENT_OUTPUT_DIR,
        alignment_strategy="chunk",
        start_wildcard=True,
        end_wildcard=True,
        blank_id=0,
        word_boundary="|",
        chunk_size=30,
        ndigits=5,
        indent=2,
        save_json=True,
        save_msgpack=False,
        return_alignments=True,
        delete_emissions=False,
        remove_wildcards=True,
        device="cuda",
    )

    # Time with time
    end_time = time.time()
    logger.info(f"Alignment complete! Output saved to {ALIGNMENT_OUTPUT_DIR}")
    logger.info(f"Processed {len(alignments)} audio files with word-level alignments.")
    logger.info(f"Time taken: {end_time - start_time}")
