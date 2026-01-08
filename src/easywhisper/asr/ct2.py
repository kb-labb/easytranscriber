"""
CTranslate2-based Whisper transcription module.

This module provides a transcribe function using ctranslate2 for efficient
Whisper inference, mirroring the HuggingFace implementation in hf.py.
"""

from pathlib import Path

import ctranslate2
import numpy as np
import torch
from easyalign.utils import save_metadata_json, save_metadata_msgpack
from transformers import WhisperProcessor

from easywhisper.data.collators import transcribe_collate_fn


def transcribe(
    model: ctranslate2.models.Whisper,
    processor: WhisperProcessor,
    file_dataloader: torch.utils.data.DataLoader,
    language: str | None = None,
    batch_size: int = 8,
    task: str = "transcribe",
    beam_size: int = 5,
    patience: float = 1.0,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    max_length: int = 448,
    suppress_blank: bool = True,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    output_dir: str = "output/transcriptions",
):
    """
    Transcribe audio files using CTranslate2 Whisper model.

    This function processes audio files through a dataloader structure similar
    to the HuggingFace implementation, but uses ctranslate2 for inference.

    Args:
        model: CTranslate2 Whisper model.
        processor: WhisperProcessor for tokenization and decoding.
        file_dataloader: DataLoader yielding audio file datasets.
        batch_size: Batch size for feature processing.
        language: Language code (e.g., 'sv', 'en'). If None, auto-detect.
        task: Task type - 'transcribe' or 'translate'.
        beam_size: Beam size for search.
        patience: Beam search patience factor.
        length_penalty: Length penalty for beam search.
        repetition_penalty: Repetition penalty.
        no_repeat_ngram_size: N-gram size for no repeat.
        max_length: Maximum output length.
        suppress_blank: Whether to suppress blank tokens.
        num_workers: Number of workers for feature dataloader (file dataloader is created outside
            of this function).
        prefetch_factor: Prefetch factor for feature dataloader (file dataloader is created outside
            of this function).
        output_dir: Directory to save transcription JSON files.
    """
    for features in file_dataloader:
        slice_dataset = features[0]["dataset"]
        metadata = features[0]["dataset"].metadata
        transcription_texts = []
        language_detections = []

        feature_dataloader = torch.utils.data.DataLoader(
            slice_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            collate_fn=transcribe_collate_fn,
        )

        for batch in feature_dataloader:
            # Convert to numpy for ctranslate2
            batch_features = batch["features"].numpy()
            current_batch_size = batch_features.shape[0]

            # Convert to ctranslate2 StorageView
            features_ct2 = ctranslate2.StorageView.from_array(batch_features)

            if language is not None:
                # Build the prompt tokens for the batch
                prompt_tokens = [
                    "<|startoftranscript|>",
                    f"<|{language}|>",
                    f"<|{task}|>",
                    "<|notimestamps|>",
                ]
                prompt_ids = processor.tokenizer.convert_tokens_to_ids(prompt_tokens)
                prompt_ids = [prompt_ids] * current_batch_size
            else:
                languages = detect_language(model, features_ct2)
                language_detections.extend(languages)
                prompt_ids = []
                for language in languages:
                    prompt_tokens = [
                        "<|startoftranscript|>",
                        language["language"],
                        f"<|{task}|>",
                        "<|notimestamps|>",
                    ]

                    prompt_ids.append(processor.tokenizer.convert_tokens_to_ids(prompt_tokens))

            # Generate transcriptions
            outputs = model.generate(
                features_ct2,
                prompt_ids,
                beam_size=beam_size,
                patience=patience,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_length=max_length,
                suppress_blank=suppress_blank,
            )

            # Decode the sequences
            sequences = [result.sequences_ids[0] for result in outputs]
            transcription = processor.batch_decode(sequences, skip_special_tokens=True)

            transcription_texts.extend(transcription)

        # Update metadata with transcriptions
        for i, speech in enumerate(metadata.speeches):
            for j, chunk in enumerate(speech.chunks):
                chunk.text = transcription_texts[j]
                if len(language_detections) > 0:
                    chunk.language = language_detections[j]["language"]
                    chunk.language_probability = language_detections[j]["probability"]

        # Save transcription to file
        output_path = Path(output_dir) / Path(metadata.audio_path).with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_metadata_json(metadata, output_dir=output_dir)


def detect_language(model: ctranslate2.models.Whisper, features: ctranslate2.StorageView) -> list:
    """
    Return the highest probability language for each chunk in the features batch.
    """

    # List of tuple[str, float] with language and probability
    lang_probs = model.detect_language(features)

    top_langs = []
    for chunk in lang_probs:
        top_langs.append({"language": chunk[0][0], "probability": chunk[0][1]})

    return top_langs
