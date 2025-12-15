import torch
import gradio as gr
import tempfile
import os
import uuid
import scipy.io.wavfile
import time
import numpy as np
import torchaudio
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer, pipeline
import subprocess

import numpy as np
from RealtimeTranslator.Translator import translate
from RealtimeTranslator.Transcriber import wav2vec_transcribe

audio_buffer = np.array([], dtype=np.float32)
prev_buffer_size = 0
current_transcript = ""
prev_transcript = ""
full_transcript = ""
translation_ = ""



def reset_states():
  global audio_buffer, prev_buffer_size, current_transcript, prev_transcript, full_transcript, translation_
  audio_buffer = np.array([], dtype=np.float32)
  prev_buffer_size = 0
  current_transcript = ""
  prev_transcript = ""
  full_transcript = ""
  translation_ = ""

def stream_processer(waveform):
    global audio_buffer, prev_buffer_size, current_transcript, prev_transcript, full_transcript, translation_
    audio_buffer = np.append(audio_buffer, waveform)

    if len(audio_buffer) >= 2 * 16000:
      transcript = wav2vec_transcribe(np.copy(audio_buffer))
      translation_ = translate(transcript)
      current_transcript = prev_transcript + transcript

      if len(audio_buffer) >= 10 * 16000:
        audio_buffer  = audio_buffer[prev_buffer_size:]
        prev_transcript = current_transcript

    prev_buffer_size = len(audio_buffer)
    print(f"contains {len(audio_buffer)/16000} seconds of audio")
    return current_transcript, translation_

def stream_transcribe(stream, new_chunk):
    global stream_record
    start_time = time.time()
    try:
        sr, y = new_chunk
        print(sr)
        # Convert to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)

        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        if sr!=16000:
          tensor_samples = torch.tensor(y, dtype=torch.float32)
          y = torchaudio.functional.resample(tensor_samples, orig_freq=sr, new_freq=16_000).cpu().numpy()

        if stream is not None:
            stream = np.concatenate([stream, y])
        else:
            stream = y

        transcription = wav2vec_transcribe(stream[-10*16000:])

        translation = translate(transcription)
        # transcription = f"dummy text: {sr}"
        end_time = time.time()
        latency = end_time - start_time

        return stream, transcription, translation,  f"{latency:.2f}"
    except Exception as e:
        print(f"Error during Transcription: {e}")
        return stream, e, "Error"
    
def transcribe(inputs, previous_transcription):
    start_time = time.time()
    pipe = None
    try:
        filename = f"{uuid.uuid4().hex}.wav"
        sample_rate, audio_data = inputs
        scipy.io.wavfile.write(filename, sample_rate, audio_data)

        transcription = pipe(filename)["text"]
        previous_transcription += transcription

        end_time = time.time()
        latency = end_time - start_time
        return previous_transcription,  f"{latency:.2f}"
    except Exception as e:
        print(f"Error during Transcription: {e}")
        return previous_transcription, "Error"
       
def clear():
    return ""

def clear_state():
    return None

def setup_app():
    
    with gr.Blocks(css="""
    #output-column {
        height: 80vh;
    }
    .big-box textarea {
        font-size: 16px;
        line-height: 1.5;
    }
    """) as demo:

        gr.Markdown(
            f"""
    # 🎙️ Realtime Transcription & Translation

    Using **{"wav2vec"}**
    First token may take ~5s, after that it's real-time.
    """
        )

        with gr.Row(equal_height=True):

            # 🎛️ Controls (small)
            with gr.Column(scale=1):
                input_audio_microphone = gr.Audio(
                    streaming=True,
                    label="Microphone"
                )

                latency_textbox = gr.Textbox(
                    label="Latency (seconds)",
                    value="0.0",
                    interactive=False
                )

                clear_button = gr.Button("Clear Output")

            # 📄 Outputs (BIG – 80%)
            with gr.Column(scale=4, elem_id="output-column"):

                output = gr.Textbox(
                    label="📝 Transcription",
                    value="",
                    lines=12,
                    max_lines=12,
                    elem_classes="big-box"
                )

                translation_output = gr.Textbox(
                    label="🌍 Translation",
                    value="",
                    lines=12,
                    max_lines=12,
                    elem_classes="big-box"
                )

        state = gr.State()


        input_audio_microphone.stream(
            stream_transcribe,
            inputs=[state, input_audio_microphone],
            outputs=[state, output, translation_output, latency_textbox],
            time_limit=30,
            stream_every=2,
            concurrency_limit=None
        )

        clear_button.click(
            clear_state,
            outputs=[state]
        ).then(
            clear,
            outputs=[output, translation_output, latency_textbox]
        )
        
    return demo

def launch_app(share = True, debug = True):
    reset_states()
    demo = setup_app()
    demo.launch(share = share, debug = debug)
