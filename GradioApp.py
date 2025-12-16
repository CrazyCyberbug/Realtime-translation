import torch
import gradio as gr
import tempfile
import os
import uuid
import scipy.io.wavfile
import time
import numpy as np
import torchaudio

from transformers import logging as hf_logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer, pipeline
import subprocess
from IPython.display import clear_output
import numpy as np
import warnings

from RealtimeTranslator.Translator import translate
from RealtimeTranslator.Transcriber import transcribe

# suppress warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_LOG_LEVEL"] = "ERROR"




audio_buffer = np.array([], dtype=np.float32)
prev_buffer_size = 0

# transcription artefacts
current_transcript = ""
prev_transcript = ""
full_transcript = ""

# translation artefacts
translation_ = ""
prev_translation = ""
current_translation = ""
full_translation = ""

LANG_MAP = {
    "Bengali": "ben_Beng",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",   
}

def reset_states():
    global audio_buffer, prev_buffer_size
    global current_transcript, prev_transcript, full_transcript
    global prev_translation, current_translation, full_translation, translation_
    
    # audio artefacts
    audio_buffer = np.array([], dtype=np.float32)
    prev_buffer_size = 0
    
    # transcription artefacts
    current_transcript = ""
    prev_transcript = ""
    full_transcript = ""
    
    # translation artefacts
    prev_translation = ""
    current_translation = ""
    full_translation = ""

def stream_processer(waveform, use_whisper_turbo = True, tgt_lang = "hin_Deva"):
    
    global audio_buffer, prev_buffer_size
    global current_transcript, prev_transcript, full_transcript
    global prev_translation, current_translation, full_translation
    
    audio_buffer = np.append(audio_buffer, waveform)

    if len(audio_buffer) >= 2 * 16000:
      if use_whisper_turbo:
        transcript = transcribe(np.copy(audio_buffer), use_whisper_turbo=True)

      else:
        transcript = transcribe(np.copy(audio_buffer), use_whisper_turbo=False)

      translation_ = translate(transcript, tgt_lang)


      current_transcript = prev_transcript + transcript
      current_translation = prev_translation + translation_

      if len(audio_buffer) >= 10 * 16000:
        audio_buffer  = audio_buffer[prev_buffer_size:]
        prev_transcript = current_transcript
        prev_translation = current_translation

    prev_buffer_size = len(audio_buffer)
    print(f"contains {len(audio_buffer)/16000} seconds of audio")
    return current_transcript, current_translation

def stream_transcribe(stream, new_chunk, tgt_lang):
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

        transcription, translation = stream_processer(y, use_whisper_turbo = True, tgt_lang = tgt_lang)

        # transcription = f"dummy text: {sr}"
        end_time = time.time()
        latency = end_time - start_time

        return stream, transcription, translation,  f"{latency:.2f}"
    except Exception as e:
        print(f"Error during Transcription: {e}")
        return stream, e, "Error"
    
# def transcribe(inputs, previous_transcription):
#     start_time = time.time()
#     pipe = None
#     try:
#         filename = f"{uuid.uuid4().hex}.wav"
#         sample_rate, audio_data = inputs
#         scipy.io.wavfile.write(filename, sample_rate, audio_data)

#         transcription = pipe(filename)["text"]
#         previous_transcription += transcription

#         end_time = time.time()
#         latency = end_time - start_time
#         return previous_transcription,  f"{latency:.2f}"
#     except Exception as e:
#         print(f"Error during Transcription: {e}")
#         return previous_transcription, "Error"
       
def clear():
    return "", "", "0.0"

def clear_state():
    return None

def setup_app(MODEL_NAME = "Openai/Whisper-large-v3-turbo"):
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

                Using **{MODEL_NAME}**
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

                language_dropdown = gr.Dropdown(
                    choices=list(LANG_MAP.keys()),
                    value="Hindi",
                    label="Source Language",
                    interactive=True
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
        tgt_lang_state = gr.State("hin_Deva")

        # Update source language ONLY when stream is off
        language_dropdown.change(
            lambda lang: LANG_MAP[lang],
            inputs=language_dropdown,
            outputs=tgt_lang_state
        )

        # Streaming: lock language dropdown
        input_audio_microphone.stream(
            stream_transcribe,
            inputs=[state, input_audio_microphone, tgt_lang_state],
            outputs=[state, output, translation_output, latency_textbox],
            time_limit=None,
            stream_every=2,
            concurrency_limit=None
        ).then(
            lambda: gr.update(interactive=False),
            outputs=language_dropdown
        )

        # Clear = reset + unlock dropdown
        clear_button.click(
            lambda: (reset_states(), None)[1],
            outputs=[state]
        ).then(
            clear,
            outputs=[output, translation_output, latency_textbox]
        ).then(
            lambda: gr.update(interactive=True),
            outputs=language_dropdown
        )
        
    return demo
               
def launch_app(share = True, debug = True):
    print("Launching UI")
    reset_states()
    demo = setup_app()
    app, local_url, public_url = demo.launch(
    share=True,
    inline=False,
    prevent_thread_lock=True
    )
    
    clear_output(wait=True)
    print("start-up complete!")
    print("Public link:", public_url)
