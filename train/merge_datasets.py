# merge_datasets.py
import json
import random

def merge_datasets(preflop_file, flop_file, output_file, add_stage_marker=True):
    """
    Объединяет префлоп и флоп датасеты.
    
    Args:
        preflop_file: путь к префлоп-датасету
        flop_file: путь к флоп-датасету
        output_file: путь для сохранения объединённого датасета
        add_stage_marker: добавить метку стадии в промпт
    """
    # Загружаем датасеты
    with open(preflop_file, 'r', encoding='utf-8') as f:
        preflop_data = json.load(f)
    
    with open(flop_file, 'r', encoding='utf-8') as f:
        flop_data = json.load(f)
    
    # Добавляем метку стадии
    if add_stage_marker:
        for ex in preflop_data:
            if "stage" not in ex.get("metadata", {}):
                ex["metadata"]["stage"] = "preflop"
            # Добавляем маркер в промпт
            if not ex["prompt"].startswith("[СТАДИЯ:"):
                ex["prompt"] = f"[СТАДИЯ: ПРЕФЛОП]\n\n{ex['prompt']}"
        
        for ex in flop_data:
            if "stage" not in ex.get("metadata", {}):
                ex["metadata"]["stage"] = "flop"
            if not ex["prompt"].startswith("[СТАДИЯ:"):
                ex["prompt"] = f"[СТАДИЯ: ФЛОП]\n\n{ex['prompt']}"
    
    # Объединяем и перемешиваем
    combined = preflop_data + flop_data
    random.seed(42)
    random.shuffle(combined)
    
    # Сохраняем
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=1)
    
    print(f"✅ Объединённый датасет сохранён: {output_file}")
    print(f"   Всего примеров: {len(combined)}")
    print(f"   Префлоп: {len(preflop_data)}")
    print(f"   Флоп: {len(flop_data)}")
    
    return combined


if __name__ == "__main__":
    merge_datasets(
        preflop_file="dataset_preflop_3.json",
        flop_file="dataset_flop_v2.json",
        output_file="dataset_combined.json",
        add_stage_marker=True
    )