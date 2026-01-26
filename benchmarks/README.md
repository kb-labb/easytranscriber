## Pipeline

### Bench1

Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz  
NVIDIA GeForce RTX 3090  

Audio duration: 343 minutes

### EasyWhisper

#### Pyannote
325.74 seconds
330.38 seconds

#### Silero
447.59 seconds

### WhisperX

664.87 seconds
657.76 seconds
668.98 seconds

### Hugging Face

#### Batch size 8, num_beams=3
1487.35 seconds

### Bench2

Audio duration: 185.07 minutes

AMD Ryzen 9 5950X 16-Core Processor
AD102GL [RTX 5000 Ada Generation] GPU

#### EasyWhisper

131.26 seconds
133.04 seconds

#### WhisperX

180.42 seconds


### Bench3

414.41883333 minutes

AMD Ryzen 9 5950X 16-Core Processor
AD102GL [RTX 5000 Ada Generation] GPU

#### EasyWhisper

305.02 seconds

#### WhisperX

412.23 seconds

#### Hugging Face

448.90 seconds


#### Bench4

DGX Station A100 40GB

300.18615984916687 (transcrib_pipeline)
328.8617465496063 (transcribe_ct2_dataloader)
328.0924553871155 (transcribe_ct2_dataloader)
357.6800391674042 (transcribe_ct2_dataloader)

#### WhisperX

606.6599538326263   (transcribe_whisperx)

#### Hugging Face

527.1353778839111  (transcribe_hf)