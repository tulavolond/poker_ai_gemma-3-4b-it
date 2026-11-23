# eval_stage1.py — запуск: python eval_stage1.py [адаптер]
import os, json, re, sys, torch, time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path; HERE = Path(__file__).resolve().parent
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

BASE = "Qwen/Qwen2.5-1.5B-Instruct"; LIMIT = 60   # для быстрого замера «до» поставь 8
# tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN, trust_remote_code=True, padding_side="right")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda")
if len(sys.argv) > 1:
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, HERE / sys.argv[1])
m.eval()

ok = n = 0
with open(HERE / "stage1_eval.jsonl", encoding="utf-8") as f:
    lines = [next(f) for _ in range(LIMIT)]
try:
    for line in tqdm(lines, desc="Оценка"):
        start = time.time()
        ex = json.loads(line)
        text = tok.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            out = m.generate(**inp, max_new_tokens=300, do_sample=False, pad_token_id=tok.eos_token_id)

        ans = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = ("1" if re.search(r"Победитель:?\s*Игрок\s*1", ans, re.I) else
                "2" if re.search(r"Победитель:?\s*Игрок\s*2", ans, re.I) else
                "split" if re.search(r"ничья|сплит|поровну|split", ans, re.I) else "?")
        n += 1; ok += pred == ex["gold"]
        if pred != ex["gold"]:
            elapsed = time.time() - start
            tqdm.write(f"#{n} [ожидал {ex['gold']}, получил {pred}] {ex['messages'][1]['content'][:160]} {elapsed:.1f}s")
except KeyboardInterrupt:
    print("\n🛑 Остановка ...")
print(f"accuracy: {ok}/{n} = {ok/n*100:.0f}%")