from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def call_mlx_lm(input_text: str, model_name: Optional[str] = "google/flan-t5-base") -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/flan-t5-base")
    args = parser.parse_args()

    output_text = call_mlx_lm(args.input_text, args.model)
    print(output_text)