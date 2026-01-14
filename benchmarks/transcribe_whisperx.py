import argparse
import logging
import re
import time
from pathlib import Path

import whisperx

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def transcribe_whisperx(
    model,
    audio,
    audio_path,
    batch_size=24,
):
    logger.info(f"Transcribing {audio_path} with batch size {batch_size}")
    result = model.transcribe(
        audio, batch_size=batch_size, num_workers=0, print_progress=False, combined_progress=False
    )
    logger.info(f"Finished transcribing {audio_path}.")

    return result


def align_whisperx(
    model_align,
    audio,
    metadata,
    result,
    audio_path,
    device="cuda",
    add_whitespace=True,
):
    logger.info(f"Aligning {audio_path}.")
    if add_whitespace:
        for i, segment in enumerate(result["segments"]):
            result["segments"][i]["text"] = add_whitespace_to_punctuation(segment["text"])

    result = whisperx.align(
        transcript=result["segments"],
        model=model_align,
        align_model_metadata=metadata,
        audio=audio,
        device=device,
        return_char_alignments=False,
        print_progress=False,
        combined_progress=False,
    )
    logger.info(f"Finished aligning {audio_path}.")

    return result


def add_whitespace_to_punctuation(text):
    """
    For punctuation marks .!? or "..." that are not followed by a space
    and that are preceeded by an alphanumeric non-punctuation mark character, add a space.
    """
    text = re.sub(r"([a-zA-Z0-9åäöÅÄÖ])([.!?]|[.]{3})(?![\s.!?])", r"\1\2 ", text)
    # Trim whitespace at beginning and end
    text = text.strip()
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir",
        default="data/sv",
        type=str,
        required=False,
        help="Directory containing audio files to transcribe.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for transcription.", required=False
    )
    args = parser.parse_args()

    audio_paths = list(Path(args.audio_dir).glob("**/*"))
    model_name = "KBLab/kb-whisper-large"
    device = "cuda"
    condition_on_previous_text = False
    no_speech_threshold = 0.6
    log_prob_threshold = -1.0
    without_timestamps = True

    # Time
    start_time = time.time()

    for audio_path in audio_paths:
        audio = whisperx.load_audio(audio_path, sr=16000)

        model_whisperx = whisperx.load_model(
            whisper_arch=model_name,
            device=device,
            compute_type="float16",
            language="sv",
            asr_options={
                "beam_size": 5,
                "condition_on_previous_text": condition_on_previous_text,
                "no_speech_threshold": no_speech_threshold,
                "log_prob_threshold": log_prob_threshold,
                "without_timestamps": without_timestamps,
                "max_initial_timestamp": 1.0,
                "max_new_tokens": 200,
                "word_timestamps": False,
            },
            vad_options={"vad_onset": 0.500, "vad_offset": 0.363},
        )
        model_align, model_metadata = whisperx.load_align_model(
            language_code="sv", device="cuda", model_name="KBLab/wav2vec2-large-voxrex-swedish"
        )

        result = transcribe_whisperx(
            model=model_whisperx,
            audio=audio,
            audio_path=audio_path,
            batch_size=16,
        )

        result_aligned = align_whisperx(
            model_align=model_align,
            audio=audio,
            audio_path=audio_path,
            metadata=model_metadata,
            result=result,
        )

    end_time = time.time()
    logger.info(f"Transcribed {len(audio_paths)} files in {end_time - start_time}")
