"""
test_preflop.py - Скрипт для тестирования обученной префлоп модели.
Запуск: python test_preflop.py
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tkinter import Tk, Entry, END


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Укажите путь к сохраненному адаптеру. 
# Если вы прервали обучение через Ctrl+C, папка может называться './lora_preflop_stage2_interrupted'
# LORA_DIR = "./lora_preflop_stage2" 
LORA_DIR = "./lora_preflop_stage2_interrupted" 
if not os.path.exists(LORA_DIR) and os.path.exists(LORA_DIR + "_interrupted"):
    LORA_DIR = LORA_DIR + "_interrupted"

print(f"📦 Загрузка базовой модели: {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"🔧 Подключение LoRA весов из: {LORA_DIR}...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()  # Переводим в режим инференса (вывода)

print("✅ Модель успешно загружена и готова к тестам!")
print("=" * 60)

# ============================================================
# ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТА
# ============================================================
def ask_poker_ai(prompt_text):
    messages = [
        {"role": "user", "content": prompt_text}
    ]
    
    # Применяем оригинальный шаблон чата Qwen
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,      # Жесткие детерминированные ответы для калькулятора
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Вырезаем входные токены, оставляя только ответ модели
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

    user_input_0 = \
            """
            === СИТУАЦИЯ: ПРЕФЛОП ===
                ТВОИ КАРТЫ: A♥, K♠
                БОРД: ещё нет карт на столе
                — АНАЛИЗ ОТ СИСТЕМЫ —
                ТВОЙ СТИЛЬ: ПРОФЕССИОНАЛ
                ПРАВИЛА СТИЛЯ: Ты сбалансирован. Рейзишь с силой > 0.65, блефуешь умеренно (12%). Учитываешь позицию и текстуру борда.
                БАНК: $15, ТВОЙ СТЕК: $816
                НУЖНО КОЛЛИРОВАТЬ: $10
                Ответь СТРОГО в формате:
                АНАЛИЗ: [одно предложение с твоей логикой]
                ДЕЙСТВИЕ: [FOLD / CHECK / CALL / RAISE]
                СУММА: [число]
            """

# ============================================================
# ИНТЕРАКТИВНЫЙ ЦИКЛ ТЕСТИРОВАНИЯ
# ============================================================

# ============================================================
# ИНТЕРАКТИВНЫЙ ЦИКЛ ТЕСТИРОВАНИЯ (МНОГОСТРОЧНЫЙ ВВОД)
# ============================================================
if __name__ == "__main__":
    print("Вставьте многострочный промпт.")
    print("👉 Чтобы ОТПРАВИТЬ текст модели, нажмите ENTER на ПУСТОЙ строке.")
    print("👉 Чтобы ВЫЙТИ, введите 'exit' на пустой строке.")
    
    while True:
        try:
            print("\n📝 Вставьте ваш текст (затем Enter на пустой строке для отправки):")
            
            lines = []
            while True:
                line = input()
                if line == "":  # Если строка пустая — прекращаем сбор данных
                    break
                lines.append(line)
            
            # Объединяем все введенные строки в один большой промпт
            user_input = "\n".join(lines).strip()
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("Выход из программы.")
                break
                
            if not user_input:
                continue
                
            print("🤖 ИИ думает...")
            answer = ask_poker_ai(user_input)
            print(f"\n🤖 Ответ ИИ:\n{answer}")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nВыход...")
            break
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
