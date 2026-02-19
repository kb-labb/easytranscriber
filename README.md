<div align="center"><img width="1020" height="340" alt="image" src="https://github.com/user-attachments/assets/7f1bdf33-5161-40c1-b6a7-6f1f586e030b" /></div>


`easytranscriber` is an automatic speech recognition library built for efficient, large-scale transcription with accurate word-level timestamps. The library is backend-agnostic, featuring modular, parallelizable, pipeline components (VAD, transcription, feature/emission extraction, forced alignment), with support for both `ctranslate2` and `Hugging Face` inference backends. Notable features include:

* **GPU accelerated forced alignment**, using [Pytorch's forced alignment API](https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html). Forced alignment is based on a GPU implementation of the Viterbi algorithm ([Pratap et al., 2024](https://jmlr.org/papers/volume25/23-1318/23-1318.pdf#page=8)).
* **Parallel loading and pre-fetching of audio files** for efficient data loading and batch processing.
* **Flexible text normalization for improved alignment quality**. Users can supply custom regex-based text normalization functions to preprocess ASR outputs before alignment. A mapping from the original text to the normalized text is maintained internally. All of the applied normalizations and transformations are consequently **non-destructive and reversible after alignment**. 
* **35% to 102% faster inference compared to [`WhisperX`](https://github.com/m-bain/whisperX)**. See the [benchmarks](#benchmarks) for more details.
* Batch inference support for both wav2vec2 and Whisper models.

## Installation

### With GPU support

```bash
pip install easytranscriber --extra-index-url https://download.pytorch.org/whl/cu128
```

> [!TIP]  
> Remove `--extra-index-url` if you want a CPU-only installation.

### Using uv

When installing with [uv](https://docs.astral.sh/uv/), it will select the appropriate PyTorch version automatically (CPU for macOS, CUDA for Linux/Windows/ARM):

```bash
uv pip install easytranscriber
```

## Usage

```python

from huggingface_hub import snapshot_download

# Download Tale of Two Cities book 1 chapter 1 LibriVox audiobook recording for testing
snapshot_download(
    "Lauler/easytranscriber_tutorials",
    repo_type="dataset",
    local_dir="data/tutorials",
    allow_patterns="tale-of-two-cities_short-en/*",
    # max_workers=4,
)

tokenizer = load_tokenizer("english") # For sentence tokenization in forced alignment
audio_files = [file.name for file in Path("data/tutorials/tale-of-two-cities_short-en").glob("*")]
pipeline(
    vad_model="pyannote",
    emissions_model="facebook/wav2vec2-base-960h",
    transcription_model="distil-whisper/distil-large-v3.5",
    audio_paths=audio_files,
    audio_dir="data/tutorials/tale-of-two-cities_short-en",
    language="en",
    tokenizer=tokenizer,
    text_normalizer_fn=text_normalizer,
    cache_dir="models",
)
```



## Benchmarks

![Benchmarks](benchmarks/plots/all_speedup.png)


## Acknowledgements

`easytranscriber` is inspired by [`WhisperX`](https://github.com/m-bain/whisperX).

The forced alignment component of `easytranscriber` is based on Pytorch's forced alignment API, which implements a GPU-accelerated version of the Viterbi algorithm as described in [Pratap et al., 2024](https://jmlr.org/papers/volume25/23-1318/23-1318.pdf#page=8).