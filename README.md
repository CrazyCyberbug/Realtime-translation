# Real-Time Speech Translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CrazyCyberbug/Realtime-translation/blob/main/Live_translation.ipynb)

## What this is

This repository contains a minimal real-time speech translation pipeline built using open-source ASR and machine translation models. The focus is on **streaming behavior**, **latency**, and **incremental decoding**, rather than offline batch translation.

The application is implemented as a Gradio app and is intended to be run from a Google Colab notebook for ease of reproduction.


## Demo
https://github.com/user-attachments/assets/3eebd6f3-0f51-4af8-8e08-df0b59619135

## System overview

At a high level, the system works as follows:

1. Audio is captured from the microphone in short chunks.
2. Incoming audio is buffered and incrementally transcribed using a streaming ASR setup.
3. Partial transcripts are translated on the fly into a target language.
4. Transcription and translation outputs are updated continuously in the UI.

This mirrors the constraints of real-world speech translation systems, where both ASR and MT operate on incomplete context.

## Language support

The current setup assumes **English speech input** and supports translation into the following languages:

- Bengali  
- Hindi  
- Kannada  
- Malayalam  
- Marathi  
- Tamil  
- Telugu  

Language selection only affects the translation stage.

## ASR scope and limitations

Only English ASR is enabled by default.  
While Indic-language speech recognition can be integrated using IndicConformer models, doing so requires NVIDIA NeMo, which significantly increases installation time and complexity—particularly in Colab environments. For this reason, Indic ASR is intentionally excluded from this demo.

## Known limitations

- **Input language**  
  The system expects English speech. Non-English input will produce unreliable transcriptions and translations.

- **Speech style sensitivity**  
  Translation quality is noticeably better for structured, fluent speech (e.g., prepared talks) than for spontaneous conversational speech. This is a known limitation of real-time MT systems and not specific to this implementation.

- **Latency**  
  End-to-end latency depends on network conditions and model inference time. Since audio is streamed over the network, unstable connections will directly impact responsiveness.

- **Context fragmentation**  
  Because transcription and translation operate on partial audio segments, sentence boundaries are not always preserved, which can affect translation coherence.

## Intended use

This project is meant as:
- A reference for building streaming ASR + MT pipelines
- A testbed for analyzing latency vs. quality trade-offs
- A practical demonstration of real-time speech translation constraints







