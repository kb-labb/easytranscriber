import logging
from pathlib import Path

import ctranslate2
import torch
from easyaligner.data.collators import audiofile_collate_fn, metadata_collate_fn
from easyaligner.data.datamodel import SpeechSegment
from easyaligner.data.dataset import (
    AudioFileDataset,
    JSONMetadataDataset,
    StreamingAudioFileDataset,
)
from easyaligner.pipelines import alignment_pipeline, emissions_pipeline, vad_pipeline
from easyaligner.vad.pyannote import load_vad_model as load_pyannote_vad_model
from easyaligner.vad.silero import load_vad_model as load_silero_vad_model
from easytranscriber.asr.ct2 import transcribe as ct2_transcribe
from easytranscriber.asr.hf import transcribe as hf_transcribe
from easytranscriber.text.normalization import text_normalizer
from easytranscriber.utils import hf_to_ct2_converter
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logger = logging.getLogger(__name__)

# dispatch mapping
TRANSCRIBE_BACKENDS = {
    "ct2": ct2_transcribe,
    "hf": hf_transcribe,
}

VAD_BACKENDS = {
    "silero": load_silero_vad_model,
    "pyannote": load_pyannote_vad_model,
}


def pipeline(
    vad_model: str,
    emissions_model: str,
    transcription_model: str,
    audio_paths: list,
    audio_dir: str,
    backend: str = "ct2",
    speeches: list[list[SpeechSegment]] | None = None,
    sample_rate: int = 16000,
    chunk_size: int = 30,
    alignment_strategy: str = "chunk",
    text_normalizer_fn: callable = text_normalizer,
    tokenizer=None,
    language: str | None = None,
    task: str = "transcribe",
    beam_size: int = 5,
    max_length: int = 250,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    patience: float = 1.0,
    no_repeat_ngram_size: int = 0,
    start_wildcard: bool = False,
    end_wildcard: bool = False,
    blank_id: int | None = None,
    word_boundary: str | None = None,
    indent: int = 2,
    ndigits: int = 5,
    batch_size_files: int = 1,
    num_workers_files: int = 2,
    prefetch_factor_files: int = 2,
    batch_size_features: int = 8,
    num_workers_features: int = 4,
    streaming: bool = True,
    save_json: bool = True,
    save_msgpack: bool = False,
    save_emissions: bool = True,
    return_alignments: bool = False,
    delete_emissions: bool = False,
    output_vad_dir: str = "output/vad",
    output_transcriptions_dir: str = "output/transcriptions",
    output_emissions_dir: str = "output/emissions",
    output_alignments_dir: str = "output/alignments",
    cache_dir: str = "models",
    hf_token: str | None = None,
    device="cuda",
):
    """
    Run the full transcription pipeline (VAD -> Transcribe -> Emissions -> Align).

    Parameters
    ----------
    vad_model : str
        Voice Activity Detection model: "pyannote" or "silero".
    emissions_model : str
        Hugging Face model ID for the emissions model ("org_name/model_name").
    transcription_model : str
        Path to Hugging Face model ID for the transcription model ("org_name/model_name").
    audio_paths : list
        List of audio file paths.
    audio_dir : str
        Directory containing audio files.
    speeches : list[list[SpeechSegment]], optional
        Existing speech segments for alignment.
    backend : str, optional
        Backend to use for the transcription model: "ct2" or "hf". Default is "ct2".
    sample_rate : int, optional
        Sample rate.
    chunk_size : int, optional
        Chunk size in seconds.
    alignment_strategy : str, optional
        Alignment strategy ('speech' or 'chunk').
    text_normalizer_fn : callable, optional
        Function to normalize text before forced alignment.
    tokenizer : object, optional
        An `nltk` tokenizer or a custom callable tokenizer that
        takes a string as input and returns a list of tuples (start_char, end_char),
        marking the spans/boundaries of sentences, paragraphs, or any other
        text unit of interest.
    beam_size : int, optional
        Number of beams for beam search. Recommended: `5` for ct2 and `1` for hf
        (beam search is slow in Hugging Face transformers).
    patience : float, optional
        Patience. Only implemented in ct2.
    length_penalty : float, optional
        Length penalty for beam search. See HF [source code](https://github.com/huggingface/transformers/blob/v4.57.5/src/transformers/generation/configuration_utils.py#L194-L198) for details
    repetition_penalty : float, optional
        See HF [source code](https://github.com/huggingface/transformers/blob/v4.57.5/src/transformers/generation/configuration_utils.py#L188-L190) for details.
    max_length : int, optional
        Maximum length of generated text.
    start_wildcard : bool, optional
        Add start wildcard to forced alignment.
    end_wildcard : bool, optional
        Add end wildcard to forced alignment.
    blank_id : int | None, optional
        Blank token ID of the emissions model (generally the pad token ID).
    word_boundary : str | None, optional
        Word boundary character of the emissions model (usually "|").
    indent : int, optional
        JSON indentation.
    ndigits : int, optional
        Number of digits for rounding.
    batch_size_files : int, optional
        Batch size for files. Recommended to set to 1.
    num_workers_files : int, optional
        Number of workers for file loading.
    prefetch_factor_files : int, optional
        Prefetch factor for files.
    batch_size_features : int, optional
        Batch size for feature extraction.
    num_workers_features : int, optional
        Number of workers for feature extraction.
    streaming : bool, optional
        Use streaming mode.
    save_json : bool, optional
        Save results to JSON.
    save_msgpack : bool, optional
        Save results to MessagePack.
    save_emissions : bool, optional
        Save emissions.
    return_alignments : bool, optional
        Return alignment results.
    delete_emissions : bool, optional
        Whether to delete emissions numpy files after processing.
    output_vad_dir : str, optional
        Output directory for VAD.
    output_transcriptions_dir : str, optional
        Output directory for transcriptions.
    output_emissions_dir : str, optional
        Output directory for emissions.
    output_alignments_dir : str, optional
        Output directory for alignments.
    cache_dir : str, optional
        Cache directory for transcription and emissions models.
    hf_token : str or None, optional
        Hugging Face authentication token for gated models.
    device : str, optional
        Device to run models on. Default is `cuda`.

    Returns
    -------
    list[list[SpeechSegment]] or None
        If `return_alignments` is True, returns a list of alignment mappings for each audio file.
        Otherwise, returns `None` (the alignments are saved to disk only).
    """  # noqa: E501
    # TODO: Support msgpack throughout the pipeline
    json_paths = [Path(p).with_suffix(".json") for p in audio_paths]

    if streaming:
        DatasetClass = StreamingAudioFileDataset
    else:
        DatasetClass = AudioFileDataset

    # Load VAD model
    assert vad_model in VAD_BACKENDS, (
        f"VAD model {vad_model} not supported. Choose from {list(VAD_BACKENDS.keys())}."
    )

    vad_model_loader = VAD_BACKENDS.get(vad_model)
    if vad_model == "pyannote":
        vad_model = vad_model_loader(device=device, token=hf_token)
    else:
        vad_model = vad_model_loader()

    # Step 1: Run VAD
    vad_pipeline(
        model=vad_model,
        audio_paths=audio_paths,
        audio_dir=audio_dir,
        speeches=speeches,
        chunk_size=chunk_size,
        sample_rate=sample_rate,
        batch_size=batch_size_files,
        num_workers=num_workers_files,
        prefetch_factor=prefetch_factor_files,
        save_json=save_json,
        save_msgpack=save_msgpack,
        return_vad=False,
        output_dir=output_vad_dir,
    )

    # Step 2: Run Transcription
    transcription_args = {
        "language": language,
        "task": task,
        "beam_size": beam_size,
        "max_length": max_length,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
    }

    if backend == "ct2":
        model_path = hf_to_ct2_converter(transcription_model, cache_dir=cache_dir)
        logger.info(f"Loading CTranslate2 model from {model_path}...")
        model = ctranslate2.models.Whisper(model_path.as_posix(), device=device)
        transcription_args.update(
            {
                "patience": patience,
                "no_repeat_ngram_size": no_repeat_ngram_size,
            }
        )
    else:
        logger.info(f"Loading Hugging Face model from {transcription_model}...")
        model = WhisperForConditionalGeneration.from_pretrained(
            transcription_model, torch_dtype=torch.float16, cache_dir=cache_dir
        ).to(device)

    processor = WhisperProcessor.from_pretrained(transcription_model, cache_dir=cache_dir)
    json_dataset = JSONMetadataDataset(
        json_paths=[str(Path(output_vad_dir) / p) for p in json_paths]
    )

    file_dataset = DatasetClass(
        metadata=json_dataset,
        processor=processor,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        alignment_strategy="chunk",
    )

    file_dataloader = torch.utils.data.DataLoader(
        file_dataset,
        batch_size=batch_size_files,
        shuffle=False,
        collate_fn=audiofile_collate_fn,
        num_workers=num_workers_files,
        prefetch_factor=prefetch_factor_files,
    )

    transcribe = TRANSCRIBE_BACKENDS[backend]
    transcribe(
        model=model,
        processor=processor,
        file_dataloader=file_dataloader,
        batch_size=batch_size_features,
        num_workers=num_workers_features,
        prefetch_factor=2,
        **transcription_args,
        output_dir=output_transcriptions_dir,
    )

    # Step 3: Extract Emissions
    json_dataset = JSONMetadataDataset(
        json_paths=[str(Path(output_transcriptions_dir) / p) for p in json_paths]
    )

    model = AutoModelForCTC.from_pretrained(emissions_model, cache_dir=cache_dir).to("cuda").half()
    processor = Wav2Vec2Processor.from_pretrained(emissions_model, cache_dir=cache_dir)

    if blank_id is None:
        blank_id = processor.tokenizer.pad_token_id
    if word_boundary is None:
        word_boundary = processor.tokenizer.word_delimiter_token

    emissions_pipeline(
        model=model,
        processor=processor,
        metadata=json_dataset,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        alignment_strategy=alignment_strategy,
        batch_size_files=batch_size_files,
        num_workers_files=num_workers_files,
        prefetch_factor_files=prefetch_factor_files,
        batch_size_features=batch_size_features,
        num_workers_features=num_workers_features,
        streaming=streaming,
        save_json=save_json,
        save_msgpack=save_msgpack,
        save_emissions=save_emissions,
        return_emissions=False,
        output_dir=output_emissions_dir,
    )

    # Step 4: Perform Alignment
    json_dataset = JSONMetadataDataset(
        json_paths=[str(Path(output_emissions_dir) / p) for p in json_paths]
    )
    json_dataloader = torch.utils.data.DataLoader(
        json_dataset,
        batch_size=batch_size_files,
        shuffle=False,
        collate_fn=metadata_collate_fn,
        num_workers=num_workers_files,
        prefetch_factor=prefetch_factor_files,
    )

    alignments = alignment_pipeline(
        dataloader=json_dataloader,
        text_normalizer_fn=text_normalizer_fn,
        processor=processor,
        tokenizer=tokenizer,
        emissions_dir=output_emissions_dir,
        output_dir=output_alignments_dir,
        alignment_strategy=alignment_strategy,
        start_wildcard=start_wildcard,
        end_wildcard=end_wildcard,
        blank_id=blank_id,
        word_boundary=word_boundary,
        chunk_size=chunk_size,
        ndigits=ndigits,
        indent=indent,
        save_json=save_json,
        save_msgpack=save_msgpack,
        return_alignments=return_alignments,
        delete_emissions=delete_emissions,
        remove_wildcards=True,
        device=device,
    )

    return alignments
