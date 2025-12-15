import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_ID = "facebook/wav2vec2-base-960h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wav2vec_processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
wav2vec_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
wav2vec_model.eval()


def wav2vec_transcribe(waveform):
  """
  waveform: torch.Tensor [1, T] or [T] at 16kHz
  """

  # Processor returns CPU tensors → move to GPU
  inputs = wav2vec_processor(
      # waveform.squeeze().cpu().numpy(),
      waveform,
      sampling_rate=16000,
      return_tensors="pt",
      padding=True
  )

  inputs = {k: v.to(device) for k, v in inputs.items()}

  with torch.no_grad():
      logits = wav2vec_model(**inputs).logits

  predicted_ids = torch.argmax(logits, dim=-1)

  # Decoding happens on CPU
  transcription = wav2vec_processor.batch_decode(
      predicted_ids.cpu()
  )[0]

  return transcription.lower()



# import torch
# import nemo.collections.asr as nemo_asr

# model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.freeze() # inference mode
# model = model.to(device) # transfer model to device



# def indic_transcribe(waveform):
#   transcription = model.transcribe(np.array(waveform.copy().astype(np.float32)), batch_size=1,logprobs=False, language_id='hi')[0][0]
#   return transcription.lower()



