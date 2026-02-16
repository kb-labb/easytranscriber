## easytranscriber

`easytranscriber` provides automatic speech recognition with accurate word-level timestamps. The library is designed for efficient transcription of large audio archives. It features a modular pipeline architecture which decomposes the different stages of the inference pipeline into independent and parallelizable components (VAD, transcription, feature/emission extraction, forced alignment). The library supports both `ctranslate2` and `Hugging Face` backends for inference. Notable features include:

* **GPU accelerated forced alignment**, using [Pytorch's forced alignment API](https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html). Forced alignment is based on a GPU implementation of the Viterbi algorithm ([Pratap et al., 2024](https://jmlr.org/papers/volume25/23-1318/23-1318.pdf#page=8)).
* **Parallel loading and pre-fetching of audio files** for efficient data loading and batch processing.
* **Flexible text normalization for improved alignment quality**. Users can supply custom regex-based text normalization functions to preprocess ASR outputs before alignment. A mapping from the original text to the normalized text is maintained internally. All of the applied normalizations and transformations are consequently **non-destructive and reversible after alignment**. 
* Batch inference support for both wav2vec2 and Whisper models.
* **35% to 102% faster inference compared to [`WhisperX`](https://github.com/m-bain/whisperX)**. See the [benchmarks](#benchmarks) section below for more details.

### Benchmarks

![Benchmarks](benchmarks/plots/all_speedup.png)