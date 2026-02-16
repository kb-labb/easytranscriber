## Benchmarks

|library     |backend        |vad      |cpu           |gpu          | cpu_ghz|pipeline       |   seconds|bench  | num_beams| audio_minutes|
|:-----------|:--------------|:--------|:-------------|:------------|-------:|:--------------|---------:|:------|---------:|-------------:|
|easywhisper |ctranslate2    |pyannote |i7-9700K      |RTX 3090     |    3.60|pipeline       |  325.7400|bench1 |         5|      343.0000|
|easywhisper |ctranslate2    |pyannote |i7-9700K      |RTX 3090     |    3.60|pipeline       |  330.3800|bench1 |         5|      343.0000|
|easywhisper |huggingface    |pyannote |i7-9700K      |RTX 3090     |    3.60|pipeline       | 1487.3500|bench1 |         3|      343.0000|
|easywhisper |ctranslate2    |silero   |i7-9700K      |RTX 3090     |    3.60|pipeline       |  447.5900|bench1 |         5|      343.0000|
|whisperx    |faster-whisper |pyannote |i7-9700K      |RTX 3090     |    3.60|whisperx       |  664.8700|bench1 |         5|      343.0000|
|whisperx    |faster-whisper |pyannote |i7-9700K      |RTX 3090     |    3.60|whisperx       |  657.7600|bench1 |         5|      343.0000|
|whisperx    |faster-whisper |pyannote |i7-9700K      |RTX 3090     |    3.60|whisperx       |  668.9800|bench1 |         5|      343.0000|
|easywhisper |ctranslate2    |pyannote |Ryzen 5950X   |RTX 5000 Ada |    3.40|pipeline       |  131.2600|bench2 |         5|      185.0700|
|easywhisper |ctranslate2    |pyannote |Ryzen 5950X   |RTX 5000 Ada |    3.40|pipeline       |  133.0400|bench2 |         5|      185.0700|
|whisperx    |faster-whisper |pyannote |Ryzen 5950X   |RTX 5000 Ada |    3.40|whisperx       |  180.4200|bench2 |         5|      185.0700|
|easywhisper |ctranslate2    |pyannote |Ryzen 5950X   |RTX 5000 Ada |    3.40|pipeline       |  305.0200|bench3 |         5|      414.4188|
|easywhisper |huggingface    |pyannote |Ryzen 5950X   |RTX 5000 Ada |    3.40|pipeline       |  448.9000|bench3 |         1|      414.4188|
|whisperx    |faster-whisper |pyannote |Ryzen 5950X   |RTX 5000 Ada |    3.40|whisperx       |  412.2300|bench3 |         5|      414.4188|
|easywhisper |ctranslate2    |pyannote |AMD Rome 7742 |A100 40GB    |    2.25|ct2_dataloader |  328.8617|bench4 |         5|            NA|
|easywhisper |ctranslate2    |pyannote |AMD Rome 7742 |A100 40GB    |    2.25|ct2_dataloader |  328.0925|bench4 |         5|            NA|
|easywhisper |ctranslate2    |pyannote |AMD Rome 7742 |A100 40GB    |    2.25|ct2_dataloader |  357.6800|bench4 |         5|            NA|
|easywhisper |huggingface    |pyannote |AMD Rome 7742 |A100 40GB    |    2.25|hf             |  527.1354|bench4 |         1|            NA|
|easywhisper |ctranslate2    |pyannote |AMD Rome 7742 |A100 40GB    |    2.25|pipeline       |  300.1862|bench4 |         5|            NA|
|whisperx    |faster-whisper |pyannote |AMD Rome 7742 |A100 40GB    |    2.25|whisperx       |  606.6600|bench4 |         5|            NA|