import logging
from pathlib import Path

import torch
from easyaligner.data.dataset import AudioFileDataset
from easyaligner.utils import save_metadata_json
from easytranscriber.data.collators import transcribe_collate_fn
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)


def transcribe(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    file_dataloader: torch.utils.data.DataLoader,
    language: str | None = None,
    task: str = "transcribe",
    batch_size: int = 4,
    beam_size: int = 5,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    max_length: int = 250,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    output_dir: str = "output/transcriptions",
    device: str = "cuda",
):
    """
    Transcribe audio files using HuggingFace Whisper model.

    Parameters
    ----------
    model : transformers.WhisperForConditionalGeneration
        HuggingFace Whisper model.
    processor : transformers.WhisperProcessor
        HuggingFace Whisper processor.
    file_dataloader : torch.utils.data.DataLoader
        DataLoader yielding audio file datasets.
    language : str, optional
        Language code (e.g., 'sv', 'en'). Default is `None` (auto-detect).
    batch_size : int, optional
        Batch size for inference.
    beam_size : int, optional
        Number of beams for beam search. Default is 5.
    length_penalty : float, optional
        Length penalty. Default is 1.0.
    repetition_penalty : float, optional
        Repetition penalty. Default is 1.0.
    max_length : int, optional
        Maximum length of generated text. Default is 250.
    num_workers : int, optional
        Number of workers for feature dataloader.
    prefetch_factor : int, optional
        Prefetch factor for feature dataloader.
    output_dir : str, optional
        Directory to save transcription JSON files. Default is `output/transcriptions`.
    device : str, optional
        Device to run inference on. Default is `cuda`.
    """
    for features in tqdm(file_dataloader, desc="Transcribing audio files"):
        slice_dataset = features[0]["dataset"]
        metadata = features[0]["dataset"].metadata
        transcription_texts = []

        feature_dataloader = torch.utils.data.DataLoader(
            slice_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=transcribe_collate_fn,
        )

        logger.info(f"Transcribing {metadata.audio_path} ...")

        for batch in feature_dataloader:
            with torch.inference_mode():
                batch = batch["features"].to(device).half()
                predicted_ids = model.generate(
                    batch,
                    return_dict_in_generate=True,
                    task=task,
                    language=language,
                    output_scores=False,
                    max_length=max_length,
                    num_beams=beam_size,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    early_stopping=True,
                )

                transcription = processor.batch_decode(
                    predicted_ids["sequences"], skip_special_tokens=True
                )

                transcription_texts.extend(transcription)

        for i, speech in enumerate(metadata.speeches):
            for j, chunk in enumerate(speech.chunks):
                chunk.text = transcription_texts[j].strip()

        # Write final transcription to file with msgspec serialization
        output_path = Path(output_dir) / Path(metadata.audio_path).with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_metadata_json(metadata, output_dir=output_dir)
