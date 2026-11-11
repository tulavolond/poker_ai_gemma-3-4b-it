import ollama
import json
import os

# MODEL_NAME = "qwen2.5:3b"
# MODEL_NAME = "gemma4:31b-cloud"
# MODEL_NAME = "gemma3:4b"
# MODEL_NAME = "nemotron-3-super:cloud"
# MODEL_NAME = "gpt-oss:120b-cloud"
MODEL_NAME = "qwen-poker:latest"
# MODEL_NAME = "blissful_ishizaka_626/gemma4-cloud:latest"
OLLAMA_BASE_URL = "http://192.168.1.100:11434"
HISTORY_FILE = "chat_memory.json"

# Сколько последних сообщений хранить (user + assistant считаются вместе)
MAX_MESSAGES = 40

client = ollama.Client(host=OLLAMA_BASE_URL)


def load_history():
    """Загружает историю из файла. Если файла нет — возвращает пустой список."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"Загружена история: {len(data)} сообщений.")
                    return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"Не удалось прочитать историю ({e}). Начинаем с чистого листа.")
    return []


def save_history(messages):
    """Сохраняет историю в файл."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Не удалось сохранить историю: {e}")


def trim_history(messages, max_messages=MAX_MESSAGES):
    """
    Обрезает историю, оставляя последние max_messages сообщений.
    Важно: стараемся не разрывать пару user/assistant и не начинать
    историю с ответа ассистента (некоторые модели это не любят).

    SYSTEM_PROMPT = (
    "Ты — ассистент с памятью. Вся переписка ниже — это единый диалог. "
    "Ты ОБЯЗАН использовать информацию из предыдущих сообщений. "
    "Если пользователь называл своё имя ранее — называй его. "
    "Никогда не говори, что не помнишь имя или не имеешь памяти: "
    "вся история диалога тебе доступна в этом запросе."
    )
    def start_chat_session():
    messages = load_history()
    # Вставляем системный промпт только если истории ещё нет
    if not messages:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    ...
    """
    if len(messages) <= max_messages:
        return messages

    trimmed = messages[-max_messages:]

    # Если первое сообщение — ответ ассистента, сдвигаем на один,
    # чтобы диалог начинался с вопроса пользователя.
    if trimmed and trimmed[0]["role"] == "assistant":
        trimmed = trimmed[1:]

    return trimmed


def start_chat_session():
    messages = load_history()

    print(f"Чат запущен с моделью {MODEL_NAME} на {OLLAMA_BASE_URL}.")
    print("Для выхода введите 'exit' или 'quit'.")
    print("Для очистки памяти введите 'clear'.")
    print("-" * 50)
    # Отладка
    # print("\n--- DEBUG: отправляю в модель ---")
    # for m in messages:
        # print(f"[{m['role']}] {m['content'][:80]}")
    # print("--- END DEBUG ---\n")
# 1. Создаем словарь с настройками 
    generation_options = { 
        "temperature": 0.3, 
        "top_p": 0.5, 
        # "top_k": 40, 
        "num_predict": 1500, 
        # "seed": 42, 
        # Для воспроизводимости результата 
        # "stop": ["\n\n", "Вы:"], 
        # # Остановка на пустой строке или обращении 
        # "presence_penalty": 0.2 
        }
    print("\n📝 Вставьте ваш текст (затем Enter на пустой строке для отправки):")
    
    while True:
        try:
            # user_input = input("Вы: ").strip()

            # if user_input.lower() in ["exit", "quit", "выход"]:
            #     save_history(messages)
            #     print("История сохранена. Завершение чата.")
            #     break

            # if user_input.lower() in ["clear", "очистить"]:
            #     messages = []
            #     save_history(messages)
            #     print("Память очищена.")
            #     continue

            # if not user_input:
            #     continue
                        
            lines = []
            while True:
                line = input("Вы: ").strip()
                if line == "":  # Если строка пустая — прекращаем сбор данных
                    break
                lines.append(line)
            
            # Объединяем все введенные строки в один большой промпт
            user_input = "\n".join(lines).strip()
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("Выход из программы.")
                break
            
            if user_input.lower() in ["clear", "очистить"]:
                messages = []
                save_history(messages)
                print("Память очищена.")
                continue
                            
            if not user_input:
                continue
            print("🤖 ИИ думает...")
            
            messages.append({"role": "user", "content": user_input})
            messages = trim_history(messages)

            response = client.chat(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
                options={
                  "temperature": 0.0,
                  "top_p": 0.1,
                  "top_k": 10,
                  "repeat_penalty": 1.15,
                  "num_ctx": 4096,
                  "num_predict": 512
                },
            )

            assistant_reply = ""
            for chunk in response:
                piece = chunk["message"]["content"]
                print(piece, end="", flush=True)
                assistant_reply += piece
            print()

            messages.append({"role": "assistant", "content": assistant_reply})
            messages = trim_history(messages)

            # Сохраняем после каждого обмена — чтобы не потерять при аварийном закрытии
            save_history(messages)

        except KeyboardInterrupt:
            save_history(messages)
            print("\nЧат прерван. История сохранена.")
            break
        except Exception as e:
            print(f"\nПроизошла ошибка: {e}")
            save_history(messages)
            break


if __name__ == "__main__":
    start_chat_session()