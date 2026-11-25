import ollama
import json
import os
import time
import requests

# РЕКОМЕНДУЕТСЯ использовать локальную модель без суффикса -cloud
# чтобы не зависеть от интернета и не платить за токены.
MODEL_NAME = "qwen2.5:3b"  
OLLAMA_BASE_URL = "http://192.168.1.100:11434"
HISTORY_FILE = "chat_history.json"
HISTORY_FILE_BROKEN = "chat_history.json.bak" # Для битых файлов

MAX_MESSAGES = 40

client = ollama.Client(host=OLLAMA_BASE_URL)


def check_internet_for_cloud():
    """Проверка интернета, если используется облачная модель."""
    if "-cloud" in MODEL_NAME or "cloud:" in MODEL_NAME:
        try:
            # Пытаемся достучаться до внешнего мира
            requests.get("https://google.com", timeout=3)
            return True
        except Exception:
            print("ОШИБКА: Выбран облачная модель, но нет доступа в интернет.")
            print("Либо включите интернет, либо смените MODEL_NAME на локальную (например, qwen:4b).")
            return False
    return True


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"Загружена история: {len(data)} сообщений.")
                    return data
        except (json.JSONDecodeError, OSError) as e:
            # Если файл битый, сохраняем его на всякий случай
            if os.path.exists(HISTORY_FILE_BROKEN):
                os.remove(HISTORY_FILE_BROKEN)
            os.rename(HISTORY_FILE, HISTORY_FILE_BROKEN)
            print(f"Файл истории поврежден ({e}). Создана резервная копия. Начинаем заново.")
    return []


def save_history(messages):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Не удалось сохранить историю: {e}")


def trim_history(messages, max_messages=MAX_MESSAGES):
    if len(messages) <= max_messages:
        return messages
    trimmed = messages[-max_messages:]
    if trimmed and trimmed[0]["role"] == "assistant":
        trimmed = trimmed[1:]
    return trimmed


def start_chat_session():
    if not check_internet_for_cloud():
        return

    messages = load_history()
    print(f"Чат запущен с моделью {MODEL_NAME} на {OLLAMA_BASE_URL}.")
    print("Введите 'exit' для выхода, 'clear' для очистки памяти.")
    print("-" * 50)
    print(messages)

    while True:
        try:
            user_input = input("Вы: ").strip()

            if user_input.lower() in ["exit", "quit", "выход"]:
                save_history(messages)
                print("История сохранена. Пока!")
                break

            if user_input.lower() in ["clear", "очистить"]:
                messages = []
                save_history(messages)
                print("Память очищена.")
                continue

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            messages = trim_history(messages)

            # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ: Обработка ошибки подключения ---
            attempt = 0
            while attempt < 2:
                try:
                    response = client.chat(
                        model=MODEL_NAME,
                        messages=messages,
                        stream=True,
                    )
                    break # Если запрос прошел, выходим из цикла попыток
                except Exception as e:
                    if "[WinError 10061]" in str(e) and attempt == 0:
                        print("\nМодель 'спит', ждем 10 секунд для загрузки...")
                        time.sleep(10)
                        attempt += 1
                        continue
                    else:
                        raise e  # Если ошибка другая или 2-я попытка не удалась

            assistant_reply = ""
            for chunk in response:
                piece = chunk["message"]["content"]
                print(piece, end="", flush=True)
                assistant_reply += piece
            print()

            messages.append({"role": "assistant", "content": assistant_reply})
            messages = trim_history(messages)
            save_history(messages)

        except KeyboardInterrupt:
            save_history(messages)
            print("\nЧат прерван. История сохранена.")
            break
        except Exception as e:
            print(f"\nКритическая ошибка: {e}")
            save_history(messages)
            break


if __name__ == "__main__":
    start_chat_session()