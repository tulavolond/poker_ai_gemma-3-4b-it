# eval_fast.py — универсальный батчевый эвалуатор для всех этапов
# Проверяем Stage 2 (Пот-оддсы)
# python eval_fast.py ./dataset_21_11/stage2/qwen25-bet-lora ./dataset_21_11/stage2/test_bet.jsonl 4

# Проверяем Stage 3 (Префлоп)
# python eval_fast.py ./dataset_21_11/stage3/qwen25-preflop-lora ./dataset_21_11/stage3/test_preflop.jsonl 4

import sys, json, re, torch, gc
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Настройки батчинга (для GTX 1070 8GB)
BATCH_SIZE = 4
MAX_NEW_TOKENS = 128

def normalize_answer(text, gold):
    """Извлекает ответ из <action> или <definition> и нормализует под формат gold."""
    m_act = re.search(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
    m_def = re.search(r"<definition>(.*?)</definition>", text, re.DOTALL | re.IGNORECASE)
    
    raw = ""
    if m_act: raw = m_act.group(1).strip().lower()
    elif m_def: raw = m_def.group(1).strip().lower()
    else: return "unknown_format"
    
    # Логика для Stage 1 (Сравнение рук: 1, 2, split)
    if gold in ["1", "2", "split"]:
        if "ничья" in raw or "split" in raw: return "split"
        if "игрок 1" in raw or raw.startswith("1"): return "1"
        if "игрок 2" in raw or raw.startswith("2"): return "2"
        return "other"
        
    # Логика для Stage 2 и 3 (Решения: fold, call, raise)
    if "фолд" in raw or "fold" in raw: return "fold"
    if "колл" in raw or "call" in raw: return "call"
    # if "рейз" in raw or "raise" in raw: return "raise"
    if "рейз" in raw or "raise" in raw or "3-бет" in raw or "3бет" in raw: return "raise"
    
    return "other"

def main():
    if len(sys.argv) < 3:
        print("Использование: python eval_fast.py <путь_к_адаптеру> <путь_к_датасету> [batch_size]")
        sys.exit(1)
        
    adapter_path = sys.argv[1]
    data_path = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else BATCH_SIZE
    
    print(f"[ADAPTER] {adapter_path}")
    print(f"[EVAL] {data_path} (batch_size={batch_size})")
    
    # Загрузка модели и адаптера
    tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", 
        dtype=torch.float16, 
        device_map="cuda"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path).eval()
    
    # Критически важно для батчевой генерации (left-padding)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.config.pad_token_id = tok.pad_token_id
    
    # Чтение датасета
    rows = [json.loads(l) for l in open(data_path, encoding="utf-8")]
    
    # Динамический сбор статистики по всем доступным ключам
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    stats_class = defaultdict(lambda: [0, 0])
    errors = []
    format_errors = 0
    
    # Батчевый цикл
    for i in tqdm(range(0, len(rows), batch_size), desc="Оценка"):
        batch_rows = rows[i:i+batch_size]
        texts = []
        
        for ex in batch_rows:
            msgs = ex["messages"]
            text = tok.apply_chat_template(
                [{"role": "system", "content": msgs[0]["content"]},
                 {"role": "user", "content": msgs[1]["content"]}],
                tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
        
        # Токенизация батча с паддингом
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=False, 
                pad_token_id=tok.eos_token_id
            )
            
        # Отсекаем входные токены (слева паддинг + промпт)
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        
        # Декодируем ответы
        responses = tok.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Анализ результатов
        for ex, resp in zip(batch_rows, responses):
            pred = normalize_answer(resp, ex["gold"])
            gold = ex["gold"]
            
            if pred == "unknown_format":
                format_errors += 1
                
            # Обновляем классы
            stats_class[gold][1] += 1
            if pred == gold:
                stats_class[gold][0] += 1
            
            # Обновляем динамические метрики (pos, cat, street, rel, bucket и т.д.)
            for key in ex.keys():
                if key not in ["messages", "gold", "task"]:
                    val = str(ex[key])
                    stats[key][val][1] += 1
                    if pred == gold:
                        stats[key][val][0] += 1
                        
            if pred != gold:
                user_prompt = ex["messages"][1]["content"][:60].replace('\n', ' ')
                errors.append(f"[{gold}!={pred}] {user_prompt}...")
                
        # Очистка VRAM между батчами (важно для 8GB карт)
        del inputs, outputs, generated_tokens
        torch.cuda.empty_cache()

    # Вывод результатов
    print("\n" + "="*50)
    print("=== ИТОГ ===")
    print("="*50)
    
    # Печатаем все найденные разрезы статистики
    for key, values in sorted(stats.items()):
        print(f"\nПо параметру '{key}':")
        for val, (c, t) in sorted(values.items()):
            pct = (c/t*100) if t > 0 else 0
            print(f"  {key} {val:<15} {c}/{t} = {pct:.0f}%")
            
    print("\nПо классам (recall):")
    for k, (c, t) in sorted(stats_class.items()):
        print(f"  {k:<10} {c}/{t} = {c/t*100:.0f}%")
        
    total_c = sum(v[0] for v in stats_class.values())
    total_t = sum(v[1] for v in stats_class.values())
    print(f"\nОбщая точность: {total_c}/{total_t} = {total_c/total_t*100:.2f}%")
    print(f"Нераспознано формата: {format_errors}/{total_t}")
    
    if errors:
        print(f"\nПервые ошибки ({len(errors)} всего):")
        for e in errors[:10]:
            print(e)
    else:
        print("\n🏆 Ошибок нет! Идеально.")

if __name__ == "__main__":
    main()