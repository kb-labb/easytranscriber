"""
Streaming audio dataset module for memory-efficient chunk-based audio loading.

This module provides datasets that read audio chunks on-demand using ffmpeg seek,
rather than loading entire files into memory. Chunk boundaries are derived from
existing metadata (SpeechSegment/AudioChunk start/end times).
"""

import logging
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, WhisperProcessor

from easyalign.data.datamodel import AudioMetadata
from easyalign.data.dataset import JSONMetadataDataset

logger = logging.getLogger(__name__)


def read_audio_segment(
    audio_path: str | Path,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Read a segment of audio using ffmpeg subprocess with seek.

    Uses ffmpeg's fast seek (-ss before -i) to efficiently read only the
    required segment, with resampling to the target sample rate and mono conversion.

    Args:
        audio_path: Path to the audio file.
        start_sec: Start time in seconds.
        duration_sec: Duration to read in seconds.
        sample_rate: Target sample rate for resampling.

    Returns:
        Audio data as float32 numpy array.
    """
    cmd = [
        "ffmpeg",
        "-ss",
        str(start_sec),  # Seek to position (before -i = fast seek)
        "-i",
        str(audio_path),
        "-t",
        str(duration_sec),  # Read this many seconds
        "-ar",
        str(sample_rate),  # Resample
        "-ac",
        "1",  # Mono
        "-f",
        "f32le",  # Raw float32 little-endian output
        "-loglevel",
        "error",
        "pipe:1",  # Output to stdout
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        return audio
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg error reading {audio_path}: {e.stderr.decode()}")
        raise


class StreamingAudioSliceDataset(Dataset):
    """
    Dataset that lazily loads audio chunks on-demand using ffmpeg seek.

    Unlike AudioSliceDataset which holds all features in memory, this dataset
    stores only the chunk metadata and loads audio when __getitem__ is called.

    Args:
        audio_path: Path to the audio file.
        chunk_specs: List of dicts with 'start_sec', 'end_sec', 'speech_id' keys.
        processor: Wav2Vec2Processor or WhisperProcessor for feature extraction.
        sample_rate: Target sample rate.
        metadata: AudioMetadata object to pass through.
    """

    def __init__(
        self,
        audio_path: str | Path,
        chunk_specs: list[dict],
        processor: Wav2Vec2Processor | WhisperProcessor,
        sample_rate: int = 16000,
        metadata: AudioMetadata | None = None,
    ):
        self.audio_path = str(audio_path)
        self.chunk_specs = chunk_specs
        self.processor = processor
        self.sample_rate = sample_rate
        self.metadata = metadata
        self.processor_attribute = (
            "input_values" if isinstance(processor, Wav2Vec2Processor) else "input_features"
        )

    def __len__(self):
        return len(self.chunk_specs)

    def __getitem__(self, idx):
        spec = self.chunk_specs[idx]
        start_sec = spec["start_sec"]
        end_sec = spec["end_sec"]
        duration_sec = end_sec - start_sec

        # Read only this chunk from disk
        audio = read_audio_segment(
            audio_path=self.audio_path,
            start_sec=start_sec,
            duration_sec=duration_sec,
            sample_rate=self.sample_rate,
        )

        # Convert to tensor and add batch dimension for processor
        if isinstance(self.processor, Wav2Vec2Processor):
            audio = torch.tensor(audio).unsqueeze(0)

        # Extract features
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        feature = getattr(inputs, self.processor_attribute)

        return {
            "feature": feature,
            "start_time_global": start_sec,
            "speech_id": spec["speech_id"],
        }


class StreamingAudioFileDataset(Dataset):
    """
    Streaming version of AudioFileDataset that reads audio chunks on-demand.

    Instead of loading entire audio files and chunking in memory, this dataset
    returns a StreamingAudioSliceDataset that lazily loads each chunk via ffmpeg.

    Args:
        metadata: List of AudioMetadata objects, JSONMetadataDataset, or single AudioMetadata.
        processor: Wav2Vec2Processor or WhisperProcessor for feature extraction.
        audio_dir: Base directory for audio files.
        sample_rate: Target sample rate for resampling.
        chunk_size: Maximum chunk size in seconds (for speech-based chunking).
        alignment_strategy: 'speech' or 'chunk' - determines how chunks are defined.
    """

    def __init__(
        self,
        metadata: JSONMetadataDataset | list[AudioMetadata] | AudioMetadata,
        processor: Wav2Vec2Processor | WhisperProcessor,
        audio_dir: str = "data",
        sample_rate: int = 16000,
        chunk_size: int = 30,
        alignment_strategy: str = "speech",
    ):
        if isinstance(metadata, AudioMetadata):
            self.metadata = [metadata]
        else:
            self.metadata = metadata

        self.audio_dir = audio_dir
        self.sr = sample_rate
        self.chunk_size = chunk_size
        self.processor = processor
        self.alignment_strategy = alignment_strategy

    def _get_speech_chunk_specs(self, metadata: AudioMetadata) -> list[dict]:
        """
        Build chunk specs from SpeechSegments, splitting into chunk_size pieces.

        This mirrors the behavior of AudioFileDataset.get_speech_features().
        """
        chunk_specs = []
        for speech in metadata.speeches:
            speech_start = speech.start
            speech_end = speech.end
            speech_duration = speech_end - speech_start

            # Calculate audio frames for the speech segment
            speech.audio_frames = int(speech_duration * self.sr)

            # Split into chunk_size sized pieces
            offset = 0.0
            while offset < speech_duration:
                chunk_start = speech_start + offset
                chunk_end = min(chunk_start + self.chunk_size, speech_end)

                chunk_specs.append(
                    {
                        "start_sec": chunk_start,
                        "end_sec": chunk_end,
                        "speech_id": speech.speech_id,
                    }
                )
                offset += self.chunk_size

        return chunk_specs

    def _get_vad_chunk_specs(self, metadata: AudioMetadata) -> list[dict]:
        """
        Build chunk specs from existing VAD chunks in metadata.

        This mirrors the behavior of AudioFileDataset.get_vad_features().
        """
        chunk_specs = []
        for speech in metadata.speeches:
            for vad_chunk in speech.chunks:
                start_sec = vad_chunk.start
                end_sec = vad_chunk.end

                # Calculate audio frames for the chunk
                vad_chunk.audio_frames = int((end_sec - start_sec) * self.sr)

                chunk_specs.append(
                    {
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "speech_id": speech.speech_id,
                    }
                )

        return chunk_specs

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        metadata = self.metadata[idx]
        audio_path = Path(self.audio_dir) / metadata.audio_path

        # Assign speech IDs if missing
        for i, speech in enumerate(metadata.speeches):
            if speech.speech_id is None:
                speech.speech_id = i

        logger.info(f"Creating streaming dataset for {audio_path}")

        # Build chunk specs based on alignment strategy
        if self.alignment_strategy == "chunk":
            chunk_specs = self._get_vad_chunk_specs(metadata)
        else:
            chunk_specs = self._get_speech_chunk_specs(metadata)

        # Return a streaming dataset for the inner dataloader
        slice_dataset = StreamingAudioSliceDataset(
            audio_path=audio_path,
            chunk_specs=chunk_specs,
            processor=self.processor,
            sample_rate=self.sr,
            metadata=metadata,
        )

        return {
            "dataset": slice_dataset,
            "audio_path": metadata.audio_path,
        }
