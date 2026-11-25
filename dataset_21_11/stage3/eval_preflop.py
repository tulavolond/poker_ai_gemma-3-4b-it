# eval_preflop.py — оценка адаптера для Stage 3 (Префлоп-чарты)
# Используй eval_fast.py для ускорения х4
# Проверяем Stage 2 (Пот-оддсы)
# python eval_fast.py ./dataset_21_11/stage2/qwen25-bet-lora ./dataset_21_11/stage2/test_bet.jsonl 4

# Проверяем Stage 3 (Префлоп)
# python eval_fast.py ./dataset_21_11/stage3/qwen25-preflop-lora ./dataset_21_11/stage3/test_preflop.jsonl 4

import sys, json, re, torch, time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

HERE = Path(__file__).resolve().parent

def extract_action(text):
    """Извлекает действие из тега <action> и нормализует его."""
    m = re.search(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return "unknown"
    
    act = m.group(1).strip().lower()
    if "рейз" in act or "raise" in act:
        return "raise"
    if "фолд" in act or "fold" in act:
        return "fold"
    return "unknown"

def main():
    if len(sys.argv) < 2:
        print("Использование: python eval_preflop.py <путь_к_адаптеру> [путь_к_датасету]")
        print("Пример: python eval_preflop.py qwen25-preflop-lora dev_preflop.jsonl")
        sys.exit(1)
        
    adapter_path = sys.argv[1]
    data_path = HERE / sys.argv[2] if len(sys.argv) > 2 else str(HERE / "dev_preflop.jsonl")
    
    print(f"\n[ADAPTER] {adapter_path}")
    print(f"[EVAL] {data_path}")
    
    # Загрузка модели и адаптера
    tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", 
        dtype=torch.float16,  # Pascal: fp16
        device_map="cuda"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path).eval()
    
    # Чтение датасета
    rows = [json.loads(l) for l in open(data_path, encoding="utf-8")]
    
    stats_pos = defaultdict(lambda: [0, 0])   # [correct, total] по позициям
    stats_cat = defaultdict(lambda: [0, 0])   # [correct, total] по категориям
    stats_class = defaultdict(lambda: [0, 0]) # [correct, total] по классам (raise/fold)
    errors = []
    format_errors = 0
    n = 0
    
    for ex in tqdm(rows, desc="Оценка"):
        start = time.time(); n=+1
        msgs = ex["messages"]
        system = msgs[0]["content"]
        user = msgs[1]["content"]
        
        # Формируем промпт для генерации
        text = tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True
        )
        inp = tok(text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            out = model.generate(
                **inp, 
                max_new_tokens=256, 
                do_sample=False, 
                pad_token_id=tok.eos_token_id
            )
            
        resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        
        pred = extract_action(resp)
        gold = ex["gold"] # "raise" или "fold"
        pos = ex["pos"]
        cat = ex["cat"]
        
        if pred == "unknown":
            format_errors += 1
            
        # Статистика
        stats_pos[pos][1] += 1
        stats_cat[cat][1] += 1
        stats_class[gold][1] += 1
        
        if pred == gold:
            stats_pos[pos][0] += 1
            stats_cat[cat][0] += 1
            stats_class[gold][0] += 1
        else:
            errors.append(f"[{pos}/{cat}] ждал {gold}, получил {pred} :: {user[:80]}...")
            elapsed = time.time() - start
            tqdm.write(f"#{n} [ожидал {ex['gold']}, получил {pred}] {user[:180]} {elapsed:.1f}s")

    # Вывод результатов
    print("\n=== ИТОГ ===")
    
    print("\nПо позициям:")
    for k, (c, t) in sorted(stats_pos.items()):
        print(f"  pos {k:<10} {c}/{t} = {c/t*100:.0f}%")
        
    print("\nПо категориям рук:")
    for k, (c, t) in sorted(stats_cat.items()):
        print(f"  cat {k:<20} {c}/{t} = {c/t*100:.0f}%")
        
    print("\nПо классам (recall):")
    for k, (c, t) in sorted(stats_class.items()):
        print(f"  {k:<10} {c}/{t} = {c/t*100:.0f}%")
        
    total_c = sum(v[0] for v in stats_pos.values())
    total_t = sum(v[1] for v in stats_pos.values())
    print(f"\nОбщая точность: {total_c}/{total_t} = {total_c/total_t*100:.2f}%")
    print(f"Нераспознано формата (<action>): {format_errors}/{total_t}")
    
    if errors:
        print("\nПервые ошибки:")
        for e in errors[:10]:
            print(e)
    else:
        print("\nОшибок нет! Идеально.")

if __name__ == "__main__":
    main()