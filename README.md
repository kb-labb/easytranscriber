<div align="center"><img width="1020" height="340" alt="image" src="https://github.com/user-attachments/assets/7f1bdf33-5161-40c1-b6a7-6f1f586e030b" /></div>


`easytranscriber` is an automatic speech recognition library built for efficient, large-scale transcription with accurate word-level timestamps. The library is backend-agnostic, featuring modular, parallelizable, pipeline components (VAD, transcription, feature/emission extraction, forced alignment), with support for both `ctranslate2` and `Hugging Face` inference backends. Notable features include:

* **GPU accelerated forced alignment**, using [Pytorch's forced alignment API](https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html). Forced alignment is based on a GPU implementation of the Viterbi algorithm ([Pratap et al., 2024](https://jmlr.org/papers/volume25/23-1318/23-1318.pdf#page=8)).
* **Parallel loading and pre-fetching of audio files** for efficient data loading and batch processing.
* **Flexible text normalization for improved alignment quality**. Users can supply custom regex-based text normalization functions to preprocess ASR outputs before alignment. A mapping from the original text to the normalized text is maintained internally. All of the applied normalizations and transformations are consequently **non-destructive and reversible after alignment**. 
* **35% to 102% faster inference compared to [`WhisperX`](https://github.com/m-bain/whisperX)**. See the [benchmarks](#benchmarks) for more details.
* Batch inference support for both wav2vec2 and Whisper models.

### Benchmarks

![Benchmarks](benchmarks/plots/all_speedup.png)
