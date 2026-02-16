import subprocess

import ctranslate2
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperProcessor

# # Resample to wav
# subprocess.run(
#     [
#         "ffmpeg",
#         "-i",
#         "YS_sr_p1_2003-09-02_0525_0600.mp4",
#         "-ar",
#         "16000",
#         "-ac",
#         "1",
#         "-c:a",
#         "pcm_s16le",
#         "YS_sr_p1_2003-09-02_0525_0600.wav",
#     ],
#     check=True,
# )


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

results = model.generate(
    features, [prompt], return_logits_vocab=True, beam_size=1, supress_tokens=[]
)
transcription = processor.decode(results[0].sequences_ids[0])

results[0].logits[0][0].numpy()
all_logits = torch.stack([torch.as_tensor(sv, device="cuda") for sv in results[0].logits[0]])
all_logits.shape

# For all logits in a result (list of StorageViews)
all_logits = np.array(
    [np.array(sv.to_device(ctranslate2.Device.cpu)) for sv in results[0].logits[0]]
)

res_np = results[0].logits[0].numpy()
res_np.shape
res_np[0].shape
res_np[0][0:10]

res = np.array(results[0].logits[0])

res.shape
