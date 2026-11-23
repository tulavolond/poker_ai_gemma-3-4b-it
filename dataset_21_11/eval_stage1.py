# eval_stage1.py — запуск: python eval_stage1.py [адаптер]
import json, re, sys, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LIMIT = 24

HERE = Path(__file__).resolve().parent
EVAL = HERE / "stage1_eval.jsonl"
print(f"[EVAL]: {EVAL}")

tok = AutoTokenizer.from_pretrained(BASE)
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda")

if len(sys.argv) > 1:
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, sys.argv[1])
m.eval()

ok = n = 0
with open(EVAL, encoding="utf-8") as f:
    lines = [next(f) for _ in range(LIMIT)]

for line in tqdm(lines, desc="Оценка"):
    ex = json.loads(line)
    text = tok.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to("cuda")
    
    with torch.inference_mode():
        out = m.generate(**inp, max_new_tokens=300, do_sample=False, pad_token_id=tok.eos_token_id)
        
    ans = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = ("1" if re.search(r"Победитель:?\s*Игрок\s*1", ans, re.I) else
            "2" if re.search(r"Победитель:?\s*Игрок\s*2", ans, re.I) else
            "split" if re.search(r"ничья|сплит|поровну|split", ans, re.I) else "?")
    n += 1
    ok += pred == ex["gold"]
    if pred != ex["gold"]:
        tqdm.write(f"#{n} [ожидал {ex['gold']}, получил {pred}] {ex['messages'][1]['content'][:160]}")

print(f"accuracy: {ok}/{n} = {ok/n*100:.0f}%")








# import json, re, sys, torch
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForCausalLM

# BASE = "Qwen/Qwen2.5-1.5B-Instruct"; LIMIT = 24
# HERE = Path(__file__).resolve().parent
# EVAL = HERE / "stage1_eval.jsonl"
# print(f"[EVAL] {EVAL}")

# tok = AutoTokenizer.from_pretrained(BASE)
# m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda")
# if len(sys.argv) > 1:
#     from peft import PeftModel
#     ad = str(Path(sys.argv[1]).resolve())
#     print(f"[ADAPTER] {ad}")
#     m = PeftModel.from_pretrained(m, ad)
# else:
#     print("[ADAPTER] BASE (без адаптера)")
# m.eval()
# ok = n = 0
# for line in list(open(EVAL, encoding="utf-8"))[:LIMIT]:
#     ex = json.loads(line)
#     text = tok.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True)
#     inp = tok(text, return_tensors="pt").to("cuda")
#     out = m.generate(**inp, max_new_tokens=300, do_sample=False)
#     ans = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
#     pred = ("1" if re.search(r"Победитель:?\s*Игрок\s*1", ans, re.I) else
#             "2" if re.search(r"Победитель:?\s*Игрок\s*2", ans, re.I) else
#             "split" if re.search(r"ничья|сплит|поровну|split", ans, re.I) else "?")
#     n += 1; ok += pred == ex["gold"]
#     if pred != ex["gold"]:
#         print(f"{index} [ожидал {ex['gold']}, получил {pred}] {ex['messages'][1]['content'][:160]}")
#         index+=1
# print(f"accuracy: {ok}/{n} = {ok/n*100:.0f}%")