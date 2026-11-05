import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "google/gemma-3-4b-it"
LORA_PATH = "./poker_gemma3_demo_lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN,
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()


def ask(board, hand):
    instruction = "Ты — покерный калькулятор. Ответь коротко: комбинация и шанс."
    user = f"{instruction}\n\nна столе {board}, у меня {hand}"

    messages = [{"role": "user", "content": user}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()


print(ask("4♠ 5♥ 4♦ J♣ 4♣", "5♣ 5♦"))
print(ask("A♠ K♠ Q♠ J♠ 2♦", "T♠ 9♠"))
print(ask("2♣ 7♦ 9♥ J♠ A♣", "K♦ K♥"))