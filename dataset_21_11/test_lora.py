# test_lora.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("./dataset_21_11/qwen25-poker-lora")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                         dtype=torch.float16, device_map="cuda")
m = PeftModel.from_pretrained(m, "./dataset_21_11/qwen25-poker-lora").eval()

def ask(system, user):
    text = tok.apply_chat_template(
        [{"role":"system","content":system},{"role":"user","content":user}],
        tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to("cuda")
    out = m.generate(**inp, max_new_tokens=700, do_sample=False)
    print(tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True), "\n---")

print("="*80)
ask("Ты эксперт по Техасскому Холдему. Дай четкое определение.", "Что такое 'Гатшот' и 'Блокер'?")
print("="*80)
ask("Ты эксперт по Техасскому Холдему без лимита (NLH Cash Game). Всегда сначала анализируй ситуацию в теге <thought>, затем выдавай решение в теге <action>.",
    "Ты на Баттоне. Твоя рука: 2♠ 7♣. Игрок до тебя сделал огромный рейз. Твои действия и почему? В ответе покажи свою руку.")
print("="*80)
ask("Ты эксперт по Техасскому Холдему. Дай четкое определение.", "Ты на Биг Блайнде (BB). Флоп: K♣ 7♣ 2♠. Твоя рука: 7♠ 7♣. Игрок до тебя сделал огромный рейз. Твои действия и почему?  В ответе покажи свою руку.")
print("="*80)
ask("Ты эксперт по Техасскому Холдему. Дай четкое определение.", "Ты на Баттоне. Твоя рука: A♠ K♣. Игрок до тебя сделал огромный рейз. Твои действия и почему?  В ответе покажи свою руку.")