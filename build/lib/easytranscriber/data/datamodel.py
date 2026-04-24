import re

import msgspec


class WordSegment(msgspec.Struct):
    """
    Word-level alignment data.

    Attributes
    ----------
    text : str
        The aligned word's text.
    start : float
        Start time of the word in seconds.
    end : float
        End time of the word in seconds.
    score : float, optional
        Optional confidence score for the word alignment.
    """

    text: str
    start: float  # in seconds
    end: float  # in seconds
    score: float | None = None  # Optional confidence score

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}


class AudioChunk(msgspec.Struct):
    """
    Segment of audio, usually created by Voice Activity Detection (VAD).

    Attributes
    ----------
    start : float
        Start time of the chunk in seconds.
    end : float
        End time of the chunk in seconds.
    text : str, optional
        Optional text transcription for the chunk.
    duration : float, optional
        Duration of the chunk in seconds.
    audio_frames : int, optional
        Number of audio frames a chunk spans.
    num_logits : int, optional
        Number of model output logits for the chunk.
    language : str, optional
        Language code for the chunk.
    language_prob : float, optional
        Probability/confidence of the detected language.
    id : str or int, optional
        Optional unique identifier for the chunk.
    """

    start: float
    end: float
    text: str | None = None
    duration: float | None = None
    audio_frames: int | None = None
    num_logits: int | None = None
    language: str | None = None
    language_prob: float | None = None
    id: str | int | None = None

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}

    def calculate_duration(self):
        self.duration = self.end - self.start
        return self.duration

    def __post_init__(self):
        if self.duration is None:
            self.calculate_duration()


class AlignmentSegment(msgspec.Struct):
    """
    A segment of aligned audio and text.

    This can be sentence, paragraph, or any other unit of text.

    Attributes
    ----------
    start : float
        Start time of the aligned segment in seconds.
    end : float
        End time of the aligned segment in seconds.
    text : str
        The aligned text segment.
    words : list[WordSegment]
        List of word-level alignment data within this segment.
    id : str or int, optional
        Optional unique identifier for the alignment segment.
    duration : float, optional
        Duration of the aligned segment in seconds.
    score : float, optional
        Optional confidence score for the alignment.
    """

    start: float  # in seconds
    end: float  # in seconds
    text: str
    words: list[WordSegment] = []
    id: str | int | None = None
    duration: float | None = None  # in seconds
    score: float | None = None  # Optional confidence score

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}

    def calculate_duration(self):
        self.duration = self.end - self.start
        return self.duration

    def __post_init__(self):
        if self.duration is None:
            self.calculate_duration()


class SpeechSegment(msgspec.Struct):
    """
    A slice of the audio file that contains speech of interest to be aligned.

    A `SpeechSegment` may be a speech given by a single speaker, a dialogue between
    multiple speakers, a book chapter, or whatever unit of organisational abstraction
    the user prefers.

    If no SpeechSegment is defined, one will automatically be added, treating the entire
    audio as a single speech.

    Attributes
    ----------
    start : float, optional
        Start time of the speech segment in seconds.
    end : float, optional
        End time of the speech segment in seconds.
    text : str, optional
        Optional text transcription (manual, or created by ASR).
    text_spans : list[tuple[int, int]], optional
        Optional (start_char, end_char) indices in the `text` that allows for a custom
        segmentation of the text to be aligned to audio. Can for example be used to
        perform alignment on paragraph, sentence, or other optional levels of granularity.
    chunks : list[AudioChunk]
        Audio chunks from which we create w2v2 logits (if `alignment_strategy` is 'chunk').
        When ASR is used, these chunks will additionally contain the transcribed text of
        the chunk. The ASR output will be used for forced alignment within the chunk.
    alignments : list[AlignmentSegment]
        Aligned text segments.
    duration : float, optional
        Duration of the speech segment in seconds.
    audio_frames : int, optional
        Number of audio frames speech segment spans.
    speech_id : str or int, optional
        Optional unique identifier for the speech segment.
    probs_path : str, optional
        Path to saved wav2vec2 emissions/probs.
    metadata : dict, optional
        Optional extra metadata such as speaker name, etc.
    """

    speech_id: str | int | None = None
    start: float | None = None  # in seconds
    end: float | None = None  # in seconds
    text: str | None = None
    text_spans: list[tuple[int, int]] | None = None
    chunks: list[AudioChunk] = []
    alignments: list[AlignmentSegment] = []  # Aligned text segments
    duration: float | None = None  # in seconds
    audio_frames: int | None = None
    probs_path: str | None = None
    metadata: dict | None = None

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}

    def calculate_duration(self):
        self.duration = self.end - self.start
        return self.duration

    def __post_init__(self):
        if self.duration is None and self.start is not None and self.end is not None:
            self.calculate_duration()

        # Assert text doesn't contain leading whitespace
        if self.text is not None:
            assert re.match(r"^\s+", self.text) is None, (
                "Text contains leading whitespace. Please .strip() the text before calculating "
                "text spans and before passing it to SpeechSegment."
            )


class AudioMetadata(msgspec.Struct):
    """
    Data model for the metadata of an audio file.

    Attributes
    ----------
    audio_path : str
        Path to the audio file.
    sample_rate : int
        Sample rate of the audio file.
    duration : float
        Duration of the audio file in seconds.
    id : str or int, optional
        Optional unique identifier for the document.
    speeches : list[SpeechSegment], optional
        List of speech segments in the audio.
    metadata : dict, optional
        Optional extra metadata.
    """

    audio_path: str
    sample_rate: int
    duration: float  # in seconds
    id: str | int | None = None
    speeches: list[SpeechSegment] | None = None  # List of speech segments in the audio
    metadata: dict | None = None  # Optional extra metadata

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}
