# eval_fast.py — python eval_fast.py <адаптер|base> [dev.jsonl] [--old-system] [--limit=16] [--dump=3]
# python ./dataset_21_11/stage1_v5/eval_fast.py ./dataset_21_11/stage1_v4/qwen25-stage1-lora dev.jsonl --old-system --limit=16 --dump=3
# (без --old-system — v5 учится с новым промптом).
# python ./dataset_21_11/stage1_v5/eval_fast.py ./dataset_21_11/stage1_v5/qwen25-stage1-lora dev.jsonl

import json, re, sys, torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
OLD_SYSTEM = "Ты эксперт по Техасскому Холдему. Твои ответы должны строго соответствовать официальным правилам оценки комбинаций."
BATCH, MAXNEW = 8, 160 #512

flags = {a.split("=")[0]: (a.split("=")[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
ADAPTER = pos[0] if pos else "base"
FILE = pos[1] if len(pos) > 1 else "dev.jsonl"
LIMIT, DUMP, OLD = int(flags.get("--limit", 0)), int(flags.get("--dump", 3)), "--old-system" in flags

tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda")
if ADAPTER != "base":
    from peft import PeftModel
    ad = str(Path(ADAPTER).resolve()); print("[ADAPTER]", ad)
    m = PeftModel.from_pretrained(m, ad)
else:
    print("[ADAPTER] BASE")
m.eval()
print("[EVAL]", HERE / FILE, "| old_system:", OLD, "| maxnew:", MAXNEW)
items = [json.loads(l) for l in open(HERE / FILE, encoding="utf-8")]
if LIMIT: items = items[:LIMIT]

def predict(ex, ans):
    a = ans.replace("**", "").replace("—", "-").replace("–", "-")
    t = ex["task"]
    if t == "compare":
        h = re.findall(r"Победител[ьяь][^\n]{0,20}?Игрок\s*([12])", a, re.I)
        if h: return h[-1]
        h = re.findall(r"Игрок\s*([12])\s*(?:выигрывает|забирает|побеждает)", a, re.I)
        if h: return h[-1]
        return "split" if re.search(r"ничья|сплит|split|поровну|делится", a, re.I) else "?"
    if t == "flush":
        if re.search(r"флеша\s*нет|нет\s*флеша|флеш\s*отсутствует", a, re.I): return "no"
        return "yes" if re.search(r"флеш\s*:|флеш\s*есть|есть\s*флеш", a, re.I) else "?"
    if re.search(r"стрита\s*нет|нет\s*стрита|стрит\s*отсутствует", a, re.I): return "no"
    return "yes" if re.search(r"стрит\s*до|стрит\s*есть|есть\s*стрит|стрит\s*:", a, re.I) else "?"

stats = defaultdict(lambda: [0, 0]); errs = []; raws = []; shown = 0
for i in tqdm(range(0, len(items), BATCH), desc="Оценка"):
    chunk = items[i:i+BATCH]
    texts = []
    for ex in chunk:
        msgs = ex["messages"][:2]
        if OLD: msgs = [{"role": "system", "content": OLD_SYSTEM}, msgs[1]]
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    inp = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        out = m.generate(**inp, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = out[:, inp["input_ids"].shape[1]:]
    for ex, ans in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
        p = predict(ex, ans); good = p == ex["gold"]
        raws.append({"task": ex["task"], "bucket": ex["bucket"], "gold": ex["gold"], "pred": p, "raw": ans})
        for key in (("task", ex["task"]), ("profile", ex["profile"]), ("bucket", ex["bucket"])):
            stats[key][0] += good; stats[key][1] += 1
        if not good:
            if p == "?" and shown < DUMP:
                shown += 1
                print(f"\n--- СЫРОЙ ОТВЕТ [{ex['task']}/{ex['bucket']}] ждал {ex['gold']} ---\n{ans[:900]}\n--- конец ---")
            if len(errs) < 15:
                errs.append(f"[{ex['task']}/{ex['bucket']}] ждал {ex['gold']}, получил {p} :: {ex['messages'][1]['content'][:90]}")

with open(HERE / f"answers_{FILE}", "w", encoding="utf-8") as f:
    for r in raws: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("\n=== ИТОГ ===")
for (kind, name), (ok, n) in sorted(stats.items()):
    print(f"{kind:8} {name:15} {ok}/{n} = {ok/n*100:.0f}%")
print(f"\nнераспознано формата: {sum(1 for r in raws if r['pred']=='?')}/{len(raws)}  (сырые ответы: answers_{FILE})")
print("\nПервые ошибки:\n" + "\n".join(errs))