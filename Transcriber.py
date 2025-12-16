import torch
import torchaudio
from enum import Enum
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer, pipeline


MODEL_ID = "facebook/wav2vec2-base-960h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wav2vec_processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
wav2vec_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
wav2vec_model.eval()



class WhisperModelChoices(Enum):
    TINY = "openai/whisper-tiny"
    BASE = "openai/whisper-base"
    SMALL = "openai/whisper-small"
    MEDIUM = "openai/whisper-medium"
    LARGE = "openai/whisper-large-v3"
    TURBO = "openai/whisper-large-v3-turbo"
    

class Wav2vec2Transcriber():
    def __init__(self, device = None):
        self.model_id = "facebook/wav2vec2-base-960h"
        available_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device if device else  available_device
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def transcribe(self, waveform):
        inputs = wav2vec_processor(waveform,
                            sampling_rate=16000,
                            return_tensors="pt",
                            padding=True)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)

        # Decoding happens on CPU
        transcription = self.processor.batch_decode(
            predicted_ids.cpu()
        )[0]

        return transcription.lower()
    
    
class WhisperTranscriber():
    def __init__(self, model_id = WhisperModelChoices.TURBO.value, device = None):
        self.model_id = model_id
        torch_dtype = torch.float16
        
        # set-up device
        available_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device if device else  available_device
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_id)
        
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id,
                                                               dtype=torch_dtype,
                                                               low_cpu_mem_usage=True,
                                                               use_safetensors=True
                                                            ).to(self.device)
                

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=10,
            dtype=torch_dtype,
            ignore_warning=True,
            device=self.device
        )
        
    def transcribe(self, waveform):
        transcription = self.pipe({"raw": waveform, "sampling_rate":16000})["text"]
        return transcription
        


whisper_transcriber = WhisperTranscriber()
wav2vec_transcriber = Wav2vec2Transcriber() 

def transcribe(waveform, use_whisper_turbo = True):
    if use_whisper_turbo:
        transcription = whisper_transcriber.transcribe(waveform)
    else:
        transcription = wav2vec_transcriber.transcribe(waveform)
    return transcription
    





