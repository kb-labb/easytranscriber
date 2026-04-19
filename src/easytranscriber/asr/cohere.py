import logging
from pathlib import Path

import torch
from easyaligner.utils import save_metadata_json
from tqdm import tqdm

from easytranscriber.data.collators import cohere_transcribe_collate_fn

logger = logging.getLogger(__name__)

# The 14 languages Cohere Transcribe was trained on.
# https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
COHERE_SUPPORTED_LANGUAGES = frozenset(
    {"ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"}
)


def _require_transformers():
    try:
        from transformers import CohereAsrForConditionalGeneration
    except ImportError as e:
        raise ImportError(
            "The 'cohere' ASR backend requires transformers>=5.4.0 "
            "(CohereAsrForConditionalGeneration is not available in the installed version). "
            "Upgrade with: pip install --upgrade 'transformers>=5.4.0'"
        ) from e
    return CohereAsrForConditionalGeneration


def transcribe(
    model,
    processor,
    file_dataloader: torch.utils.data.DataLoader,
    language: str,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    punctuation: bool = True,
    sample_rate: int = 16000,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    output_dir: str = "output/transcriptions",
    generate_kwargs: dict | None = None,
):
    """
    Transcribe audio files using the Cohere Transcribe model.

    Parameters
    ----------
    model : transformers.CohereAsrForConditionalGeneration
        Cohere ASR model.
    processor : transformers.AutoProcessor
        Cohere ASR processor.
    file_dataloader : torch.utils.data.DataLoader
        DataLoader yielding audio file datasets. The underlying
        ``StreamingAudioFileDataset`` must be constructed with
        ``return_raw_audio=True`` so the processor can be called on whole
        batches (per-sample calls return variable-length features).
    language : str
        ISO 639-1 language code (e.g. 'en', 'ja'). Required — Cohere has
        no built-in language detection.
    batch_size : int, optional
        Batch size for inference.
    max_new_tokens : int, optional
        Maximum number of tokens to generate per chunk. Default is 256.
    punctuation : bool, optional
        Emit punctuation in transcriptions. Default is True.
    sample_rate : int, optional
        Sample rate of audio passed to the processor. Default is 16000.
    num_workers : int, optional
        Number of workers for the feature dataloader.
    prefetch_factor : int, optional
        Prefetch factor for the feature dataloader.
    output_dir : str, optional
        Directory to save transcription JSON files.
    generate_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.generate()`` (e.g.
        ``num_beams``, ``length_penalty``).
    """
    _require_transformers()

    if language is None:
        raise ValueError(
            "The 'cohere' backend requires an explicit `language` — "
            "CohereAsrForConditionalGeneration does not perform language detection."
        )
    if language not in COHERE_SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Language {language!r} is not supported by Cohere Transcribe. "
            f"Supported: {sorted(COHERE_SUPPORTED_LANGUAGES)}."
        )

    generate_kwargs = generate_kwargs or {}

    for features in tqdm(file_dataloader, desc="Transcribing audio files"):
        slice_dataset = features[0]["dataset"]
        metadata = features[0]["dataset"].metadata
        transcription_texts = []

        feature_dataloader = torch.utils.data.DataLoader(
            slice_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=cohere_transcribe_collate_fn,
        )

        logger.info(f"Transcribing {metadata.audio_path} ...")

        for batch in feature_dataloader:
            inputs = processor(
                batch["audio"],
                sampling_rate=sample_rate,
                return_tensors="pt",
                language=language,
                punctuation=punctuation,
            )
            inputs = inputs.to(model.device, dtype=model.dtype)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )

                transcription = processor.batch_decode(outputs, skip_special_tokens=True)
                transcription_texts.extend(transcription)

        for i, speech in enumerate(metadata.speeches):
            for j, chunk in enumerate(speech.chunks):
                chunk.text = transcription_texts[j].strip()

        output_path = Path(output_dir) / Path(metadata.audio_path).with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_metadata_json(metadata, output_dir=output_dir)
