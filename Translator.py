import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from enum import Enum


class translator_model_choices(Enum):
    INDIC_TRANS_2 = "ai4bharat/indictrans2-en-indic-1B"
    INDIC_TRANS_DISTILLED = "ai4bharat/indictrans2-en-indic-dist-200M"


class Translator():
    """ wrapper class around transltion model. To integrate other models in future if needed
    """
    def __init__(self, model_choice = translator_model_choices.INDIC_TRANS_DISTILLED.value, device= None):
        self.model_name = model_choice
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # use cuda unless specified
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device if device else DEVICE
        
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                                    self.model_name,
                                    trust_remote_code=True,
                                    torch_dtype=torch.float16, # performance might slightly vary for bfloat16
                                ).to(self.device)

        self.ip = IndicProcessor(inference=True)
        
    def translate(self, text):
        src_lang, tgt_lang = "eng_Latn", "hin_Deva"
        
        # preprocess and tokenize
        batch = self.ip.preprocess_batch([text,], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.tokenizer(batch,truncation=True, padding="longest", 
                                return_tensors="pt", return_attention_mask=True).to(self.device)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=False,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
            
        # Decode the generated tokens into text
        generated_tokens = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # Postprocess the translations, including entity replacement
        translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        return translations
        
        
        
translator = Translator()
def translate(text):
    translations = translator.translate(text)[0]
    return translations
        
        