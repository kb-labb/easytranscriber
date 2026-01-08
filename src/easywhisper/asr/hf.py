from pathlib import Path

import torch
from easyalign.data.dataset import AudioFileDataset
from easyalign.utils import save_metadata_json
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from easywhisper.data.collators import transcribe_collate_fn


def transcribe(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    file_dataloader: torch.utils.data.DataLoader,
    language: str | None = None,
    batch_size: int = 4,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    max_length: int = 250,
    num_beams: int = 5,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    output_dir: str = "output/transcriptions",
    device: str = "cuda",
):
    for features in file_dataloader:
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

        for batch in feature_dataloader:
            with torch.inference_mode():
                batch = batch["features"].to(device).half()
                predicted_ids = model.generate(
                    batch,
                    return_dict_in_generate=True,
                    task="transcribe",
                    language=language,
                    output_scores=False,
                    max_length=max_length,
                    num_beams=num_beams,
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
                chunk.text = transcription_texts[j]

        # Write final transcription to file with msgspec serialization
        output_path = Path(output_dir) / Path(metadata.audio_path).with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_metadata_json(metadata, output_dir=output_dir)
