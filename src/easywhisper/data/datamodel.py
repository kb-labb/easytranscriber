import re

import msgspec


class WordSegment(msgspec.Struct):
    """
    Word-level alignment data.

    Attributes:
        text:
            The aligned word's text.
        start:
            Start time of the word in seconds.
        end:
            End time of the word in seconds.
        score:
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
    Segment of audio, usually created by VAD.

    Attributes:
        start: Start time of the chunk in seconds.
        end: End time of the chunk in seconds.
        text: Optional text transcription for the chunk.
        duration: Duration of the chunk in seconds.
        audio_frames: Number of audio frames a chunk spans.
        num_logits: Number of model output logits for the chunk.
        language: Optional language code for the chunk (used for routing chunks to
            language-specific models when doing ASR).
    """

    start: float
    end: float
    text: str | None = None
    duration: float | None = None
    audio_frames: int | None = None
    num_logits: int | None = None
    language: str | None = None

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

    Attributes:
        start:
            Start time of the aligned segment in seconds.
        end:
            End time of the aligned segment in seconds.
        text:
            The aligned text segment.
        words:
            List of word-level alignment data within this segment.
        duration:
            Duration of the aligned segment in seconds.
        score:
            Optional confidence score for the alignment.
    """

    start: float  # in seconds
    end: float  # in seconds
    text: str
    words: list[WordSegment] = []
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
    A slice of the audio that contains speech of interest to be aligned.

    A `SpeechSegment` may be a speech given by a single speaker, a dialogue between
    multiple speakers, a book chapter, or whatever unit of organisational abstraction
    the user prefers.

    If no SpeechSegment is defined, one will automatically be added, treating the entire
    audio as a single speech.

    Attributes:
        start:
            Start time of the speech segment in seconds.
        end:
            End time of the speech segment in seconds.
        text:
            Optional text transcription (manual, or created by ASR).
        text_spans:
            Optional (start_char, end_char) indices in the `text` that allows for a custom
            segmentation of the text to be aligned to audio. Can for example be used to
            perform alignment on paragraph, sentence, or other optional levels of granularity.
        chunks:
            Audio chunks from which we create w2v2 logits (if `alignment_strategy` is 'chunk').
            When ASR is used, these chunks will additionally contain the transcribed text of
            the chunk. The ASR output will be used for forced alignment within the chunk.
        alignments:
            Aligned text segments.
        duration:
            Duration of the speech segment in seconds.
        audio_frames:
            Number of audio frames speech segment spans.
        speech_id:
            Optional unique identifier for the speech segment.
        probs_path:
            Path to saved wav2vec2 emissions/probs.
        metadata:
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
    """

    audio_path: str
    sample_rate: int
    duration: float  # in seconds
    speeches: list[SpeechSegment] | None = None  # List of speech segments in the audio
    metadata: dict | None = None  # Optional extra metadata

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}
