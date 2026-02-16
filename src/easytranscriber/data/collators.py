import torch


def transcribe_collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for transcription.

    Parameters
    ----------
    batch : list of dict
        List of samples from the dataset.

    Returns
    -------
    dict
        Collated batch with 'features', 'start_times', and 'speech_ids'.
    """
    # Remove None values
    speech_ids = [b["speech_id"] for b in batch if b is not None]
    start_times = [b["start_time_global"] for b in batch if b is not None]
    batch = [b["feature"] for b in batch if b is not None]

    # Concat, keep batch dimension
    batch = torch.cat(batch, dim=0)

    return {
        "features": batch,
        "start_times": start_times,
        "speech_ids": speech_ids,
    }
