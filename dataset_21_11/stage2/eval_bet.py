# eval_bet.py — python eval_bet.py <адаптер|base> [dev_bet.jsonl]
import json, re, sys, torch
from pathlib import Path
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen2.5-1.5B-Instruct"; BATCH, MAXNEW = 8, 96
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
ADAPTER = pos[0] if pos else "base"
FILE = pos[1] if len(pos) > 1 else "dev_bet.jsonl"

tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda")
if ADAPTER != "base":
    from peft import PeftModel
    ad = str(Path(ADAPTER).resolve()); print("[ADAPTER]", ad)
    m = PeftModel.from_pretrained(m, ad)
m.eval(); print("[EVAL]", HERE/FILE)
items = [json.loads(l) for l in open(HERE/FILE, encoding="utf-8")]

def pred(ans):
    a = ans.replace("**","")
    mt = re.search(r"<action>\s*(Фолд|Колл)", a, re.I)
    if mt: return "fold" if mt.group(1).lower().startswith("фолд") else "call"
    if re.search(r"\bфолд\b|сброс", a, re.I): return "fold"
    if re.search(r"\bколл\b|уравн", a, re.I): return "call"
    return "?"

stats = defaultdict(lambda:[0,0]); percls = defaultdict(lambda:[0,0]); errs=[]; raws=[]
for i in tqdm(range(0,len(items),BATCH), desc="Оценка"):
    chunk = items[i:i+BATCH]
    texts = [tok.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True) for ex in chunk]
    inp = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        out = m.generate(**inp, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
    for ex, ans in zip(chunk, tok.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)):
        p = pred(ans); ok = p == ex["gold"]
        raws.append({"bucket":ex["bucket"],"gold":ex["gold"],"pred":p,"raw":ans})
        for k in (("street",ex["street"]),("profile",ex["profile"]),("rel",ex["rel"]),("bucket",ex["bucket"])):
            stats[k][0] += ok; stats[k][1] += 1
        percls[ex["gold"]][0] += ok; percls[ex["gold"]][1] += 1
        if not ok and len(errs) < 12: errs.append(f"[{ex['bucket']}] ждал {ex['gold']}, получил {p}")
with open(HERE/f"answers_{FILE}","w",encoding="utf-8") as f:
    for r in raws: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("\n=== ИТОГ ===")
for (k,n),(o,t) in sorted(stats.items()):
    if k != "bucket": print(f"{k:8} {n:12} {o}/{t} = {o/t*100:.0f}%")
print("\nпо классам (recall):", {k:f"{o}/{t}={o/t*100:.0f}%" for k,(o,t) in percls.items()})
print("предсказаний:", dict(Counter(r["pred"] for r in raws)))
print("\nошибки:\n" + "\n".join(errs))