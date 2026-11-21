# merge_lora.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                            torch_dtype=torch.float16, device_map="cpu")
m = PeftModel.from_pretrained(base, "qwen25-poker-lora").merge_and_unload()
m.save_pretrained("qwen25-poker-merged", safe_serialization=True)
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").save_pretrained("qwen25-poker-merged")