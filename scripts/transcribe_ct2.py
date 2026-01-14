import subprocess

import ctranslate2
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from transformers import WhisperProcessor

# Resample to wav
subprocess.run(
    [
        "ffmpeg",
        "-i",
        "YS_sr_p1_2003-09-02_0525_0600.mp4",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "YS_sr_p1_2003-09-02_0525_0600.wav",
    ],
    check=True,
)


audio, sample_rate = sf.read("data/sv/statsminister.wav")


# Compute the features of the first 30 seconds of audio.
processor = WhisperProcessor.from_pretrained("models/hf/kb-whisper-large")
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

model = ctranslate2.models.Whisper("models/kb-whisper-large", device="cuda")

# Detect the language.
results = model.detect_language(features)
language, probability = results[0][0]
print("Detected language %s with probability %f" % (language, probability))


# Describe the task in the prompt.
# See the prompt format in https://github.com/openai/whisper.
prompt = processor.tokenizer.convert_tokens_to_ids(
    [
        "<|startoftranscript|>",
        language,
        "<|transcribe|>",
        "<|notimestamps|>",  # Remove this token to generate timestamps.
    ]
)

results = model.generate(features, [prompt], return_logits_vocab=True, beam_size=1)
transcription = processor.decode(results[0].sequences_ids[0])

print("Transcription: %s" % transcription)

results[0].logits[0]

res = np.array(results[0].logits[0])

res.shape
