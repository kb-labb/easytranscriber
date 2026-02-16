import subprocess

import ctranslate2
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="KBLab/kb-whisper-large",
    local_dir="models/hf/kb-whisper-large",
    local_dir_use_symlinks=False,
    ignore_patterns=["onnx/*", "ggml*"],
)

# Run ct2-transformers-converter --model models/hf/kb-whisper-large --output_dir models/kb-whisper-large
# If error output to stderr, then print the error message and exit
try:
    subprocess.run(
        [
            "ct2-transformers-converter",
            "--model",
            "models/hf/kb-whisper-large",
            "--output_dir",
            "models/kb-whisper-large",
            "--copy_files",
            "tokenizer.json",
            "--copy_files",
            "preprocessor_config.json",
        ],
        check=True,
    )
except subprocess.CalledProcessError as e:
    print(f"Error during conversion: {e.stderr.decode()}")
    exit(1)
