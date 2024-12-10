from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def call_mlx_lm(input_text: str, model_name: Optional[str] = "google/flan-t5-base") -> str:
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text