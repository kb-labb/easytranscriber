import logging
import time
from pathlib import Path

import torch
from easyaligner.data.collators import (
    audiofile_collate_fn,
    metadata_collate_fn,
)
from easyaligner.data.dataset import AudioFileDataset, JSONMetadataDataset
from easyaligner.pipelines import (
    alignment_pipeline,
    emissions_pipeline,
    vad_pipeline,
)
from easyaligner.text.normalization import (
    SpanMapNormalizer,
)
from easyaligner.vad.pyannote import load_vad_model
from easytranscriber.asr.hf import transcribe
from easytranscriber.data import StreamingAudioFileDataset
from nltk.tokenize import PunktTokenizer
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def text_normalizer(text: str) -> str:
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
    model = WhisperForConditionalGeneration.from_pretrained(
        "KBLab/kb-whisper-large", torch_dtype=torch.float16
    ).to("cuda")
    processor = WhisperProcessor.from_pretrained("kblab/kb-whisper-large")
    model_vad = load_vad_model()

    audio_dir = "data/sv"
    audio_files = [file.name for file in Path(audio_dir).glob("*")]

    start_time = time.time()
    vad_outputs = vad_pipeline(
        model=model_vad,
        audio_paths=audio_files,
        audio_dir=audio_dir,
        speeches=None,
        chunk_size=30,
        sample_rate=16000,
        metadata=None,
        batch_size=1,
        num_workers=1,
        prefetch_factor=2,
        save_json=True,
        save_msgpack=False,
        output_dir="output/vad",
    )

    json_dataset = JSONMetadataDataset(json_paths=list(Path("output/vad").rglob("*.json")))

    file_dataset = StreamingAudioFileDataset(
        metadata=json_dataset,
        processor=processor,
        audio_dir=audio_dir,
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

    transcribe(
        model=model,
        processor=processor,
        file_dataloader=file_dataloader,
        batch_size=8,
        num_workers=4,
        prefetch_factor=2,
        max_length=250,
        num_beams=1,
        output_dir="output/transcriptions",
    )

    # Align with wav2vec2
    model = (
        AutoModelForCTC.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish").to("cuda").half()
    )
    processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
    json_dataset = JSONMetadataDataset(
        json_paths=list(Path("output/transcriptions").rglob("*.json"))
    )

    emissions_output = emissions_pipeline(
        model=model,
        processor=processor,
        metadata=json_dataset,
        audio_dir=audio_dir,
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
        output_dir="output/emissions",
    )

    json_dataset = JSONMetadataDataset(json_paths=list(Path("output/emissions").rglob("*.json")))
    audiometa_loader = torch.utils.data.DataLoader(
        json_dataset,
        batch_size=1,
        num_workers=2,
        prefetch_factor=2,
        collate_fn=metadata_collate_fn,
    )

    alignments = alignment_pipeline(
        dataloader=audiometa_loader,
        text_normalizer_fn=text_normalizer,
        processor=processor,
        tokenizer=None,
        emissions_dir="output/emissions",
        output_dir="output/alignments",
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
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time}")
