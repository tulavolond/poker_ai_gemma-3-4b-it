"""
Сервер модели с отладкой и обработкой ошибок.
Запуск: python model_server.py
"""
import os
import json
import traceback
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from http.server import HTTPServer, BaseHTTPRequestHandler

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
# Глобальные переменные для обучения и памяти
training_buffer = []
MEMORY_LIMIT = 50  # Обучаем только после накопления 50 примеров
last_loss = 0.0
MODEL_ID = "google/gemma-3-4b-it"
LORA_PATH = "./poker_gemma3_lora"
MEMORY_FILE = "./poker_memory.json"
PORT = 8899
HOST = "192.168.1.103"

print("=" * 60)
print("  ЗАГРУЗКА МОДЕЛИ (один раз)")
print("=" * 60)
print(f"  torch: {torch.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda device: {torch.cuda.get_device_name(0)}")
    print(f"  cuda capability: {torch.cuda.get_device_capability(0)}")

# ============================================================
# КВАНТИЗАЦИЯ — bfloat16 обязателен для Gemma-3
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # ← было float16
    bnb_4bit_use_double_quant=True,
)

# Принудительно на GPU, без "auto"
device_map = {"": 0} if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map=device_map,
    dtype=torch.bfloat16,                     # ← было float16
    token=HF_TOKEN,
    low_cpu_mem_usage=True,
)

# ============================================================
# ТОКЕНИЗАТОР
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

USE_LORA = True #*********************************************************Важно**************
if USE_LORA and os.path.exists(LORA_PATH):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, LORA_PATH)
    print("✅ LoRA подключен")
else:
    print("⚠️ LoRA отключен. Используется базовая модель.")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

# Chat template
GEMMA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\\n' }}"
    "{% else %}"
    "{{ '<start_of_turn>' + message['role'] + '\\n' }}"
    "{% endif %}"
    "{{ message['content'] + '<end_of_turn>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>model\\n' }}"
    "{% endif %}"
)

if tokenizer.chat_template is None:
    print("⚠️ chat_template отсутствует, ставлю Gemma-шаблон")
    tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

print("✅ Токенизатор загружен")
print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
print("PAD:", tokenizer.pad_token, tokenizer.pad_token_id)
print("BOS:", tokenizer.bos_token, tokenizer.bos_token_id)
print("UNK:", tokenizer.unk_token, tokenizer.unk_token_id)

end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
print("end_of_turn id:", end_of_turn_id)

# Стоп-токены: eos + end_of_turn. pad и bos — в bad_words, не в стопах
stop_ids = []
if tokenizer.eos_token_id is not None:
    stop_ids.append(tokenizer.eos_token_id)
if (
    end_of_turn_id is not None
    and end_of_turn_id != tokenizer.unk_token_id
    and end_of_turn_id != tokenizer.eos_token_id
):
    stop_ids.append(end_of_turn_id)
stop_ids = sorted(set(stop_ids))
print("STOP IDs:", stop_ids)

# Запрещённые токены: bos, pad, unk — модель не должна их генерировать
bad_ids = []
for tid in (
    tokenizer.bos_token_id,
    tokenizer.pad_token_id,
    tokenizer.unk_token_id,
):
    if tid is not None:
        bad_ids.append(tid)
bad_ids = sorted(set(bad_ids))
print("BAD IDs (нельзя генерировать):", bad_ids)

print("✅ Модель готова")


# ============================================================
# ДИАГНОСТИКА: тестовый запрос при старте
# ============================================================

def _quick_test():
    """Проверяем, что модель вообще отвечает осмысленно."""
    print("\n" + "=" * 60)
    print("  ДИАГНОСТИКА МОДЕЛИ")
    print("=" * 60)

    messages = [{"role": "user", "content": "Скажи одно слово: привет"}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            eos_token_id=stop_ids,
            bad_words_ids=[[i] for i in bad_ids],
        )

    gen = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()

    print(f"  Input ids: {inputs['input_ids'][0].tolist()}")
    print(f"  Generated ids: {gen.tolist()}")
    print(f"  Generated text: {repr(text)}")

    if not text:
        print("\n  ❌ МОДЕЛЬ НЕ ГЕНЕРИРУЕТ ТЕКСТ!")
        print("  Возможные причины:")
        print("  1. Квантизация сломана (попробуй dtype=float32 без 4-bit)")
        print("  2. Gemma-3 не поддерживается этой версией transformers")
        print("  3. Слишком мало VRAM, слои ушли на CPU")
        print("  Модель работает, но ответы будут пустыми.")
    else:
        print(f"  ✅ Модель генерирует текст\n{text}\n— сервер готов к работе.")
    print("=" * 60 + "\n")


_quick_test()


# ============================================================
# HTTP-ОБРАБОТЧИК
# ============================================================

# class ModelHandler(BaseHTTPRequestHandler):

#     def do_GET(self):
#         self.send_response(200)
#         self.send_header('Content-Type', 'application/json')
#         self.end_headers()
#         self.wfile.write(json.dumps({'status': 'ok'}).encode())

#     def do_POST(self):
#         try:
#             content_length = self.headers.get('Content-Length')
#             if content_length is None:
#                 self._send_error(400, "Нет Content-Length")
#                 return
#             length = int(content_length)
#             if length == 0:
#                 self._send_error(400, "Пустое тело")
#                 return

#             raw_body = self.rfile.read(length)
#             body = json.loads(raw_body.decode('utf-8'))

#             prompt_from_client = body.get('prompt', '')
#             max_tokens = body.get('max_tokens', 80)

#             if not prompt_from_client.strip():
#                 self._send_error(400, "Пустой prompt")
#                 return

#             messages = [{"role": "user", "content": prompt_from_client}]
#             chat_prompt = tokenizer.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True,
#             )

#             inputs = tokenizer(
#                 chat_prompt,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=2048,
#                 add_special_tokens=False,   # chat_prompt уже с <bos>
#             ).to(model.device)

#             input_length = inputs["input_ids"].shape[1]

#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     max_new_tokens=max_tokens,
#                     min_new_tokens=1,
#                     do_sample=False,
#                     repetition_penalty=1.1,
#                     eos_token_id=stop_ids,
#                     bad_words_ids=[[i] for i in bad_ids],
#                 )

#             generated_ids = outputs[0][input_length:]
#             response_text = tokenizer.decode(
#                 generated_ids, skip_special_tokens=True,
#             ).strip()

#             for tok in ('<end_of_turn>', '<eos>', '<bos>', '<pad>',
#                         '<start_of_turn>', 'model\n'):
#                 response_text = response_text.replace(tok, '')
#             response_text = response_text.strip()

#             print(f"  📥 {len(raw_body)}b → gen={generated_ids.tolist()} "
#                   f"→ {repr(response_text[:80])}", flush=True)

#             if not response_text:
#                 response_text = "ДЕЙСТВИЕ: ФОЛД | СУММА: 0"

#             self.send_response(200)
#             self.send_header('Content-Type', 'application/json')
#             self.end_headers()
#             self.wfile.write(json.dumps(
#                 {'response': response_text}, ensure_ascii=False,
#             ).encode('utf-8'))

#         except Exception as e:
#             print(f"  ❌ Ошибка: {e}", flush=True)
#             traceback.print_exc()
#             try:
#                 self._send_error(500, str(e))
#             except Exception:
#                 pass

#     def _send_error(self, code, message):
#         self.send_response(code)
#         self.send_header('Content-Type', 'application/json')
#         self.end_headers()
#         self.wfile.write(json.dumps({'error': message}).encode())

#     def log_message(self, format, *args):
#         pass

# ============================================================
# HTTP-ОБРАБОТЧИК (с CORS)
# ============================================================

# Разрешённые origins для CORS
# "*" = разрешить все (удобно для разработки)
# Для продакшена укажите конкретный origin: "http://localhost:8000"
ALLOWED_ORIGINS = "*"

# ============================================================
# НАСТРОЙКА ОБУЧЕНИЯ (LoRA Online Fine-Tuning)
# ============================================================
if USE_LORA and os.path.exists(LORA_PATH):
    # Разрешаем градиенты для LoRA
    model.enable_input_require_grads()
    # Оптимизатор только для обучаемых параметров (LoRA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    print("✅ Оптимизатор для онлайн-обучения LoRA инициализирован (lr=1e-5)")
else:
    optimizer = None
    print("⚠️ Онлайн-обучение отключено (нет LoRA или путь не найден)")

# ============================================================
# HTTP-ОБРАБОТЧИК (обновленный)
# ============================================================
ALLOWED_ORIGINS = "*"

class ModelHandler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', ALLOWED_ORIGINS)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '86400')

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok', 'lora_active': USE_LORA}).encode())

    def do_POST(self):
        """Эндпоинт для получения решения ИИ"""
        try:
            # ... (ваш существующий код чтения body и токенизации) ...
            content_length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(content_length)
            body = json.loads(raw_body.decode('utf-8'))
            
            prompt_from_client = body.get('prompt', '')
            if not prompt_from_client.strip():
                self._send_error(400, "Пустой prompt")
                return

            messages = [{"role": "user", "content": prompt_from_client}]
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(
                chat_prompt, return_tensors="pt", truncation=True, max_length=2048,
                add_special_tokens=False,
            ).to(model.device)

            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=120, do_sample=True, temperature=0.7,
                    top_p=0.9, repetition_penalty=1.1,
                    eos_token_id=stop_ids,
                    bad_words_ids=[[i] for i in bad_ids],
                )

            generated_ids = outputs[0][input_length:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Очистка от артефактов шаблона
            for tok in ('<end_of_turn>', '<eos>', '<bos>', '<pad>', '<start_of_turn>', 'model\n'):
                response_text = response_text.replace(tok, '')
            response_text = response_text.strip()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({'response': response_text}, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            print(f"  ❌ Ошибка inference: {e}", flush=True)
            traceback.print_exc()
            self._send_error(500, str(e))

    def do_PUT(self):
        """Эндпоинт для ОНЛАЙН-ОБУЧЕНИЯ (Fine-Tuning шаг)"""
        if not optimizer or not USE_LORA:
            self._send_error(501, "Обучение не настроено (нет LoRA)")
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(content_length)
            body = json.loads(raw_body.decode('utf-8'))
            
            # Ожидаем: {"prompt": "состояние игры", "completion": "правильное действие ИИ"}
            prompt = body.get('prompt', '')
            completion = body.get('completion', '')
            reward = body.get('reward', 1.0) # Множитель силы обучения (1.0 = нормально, >1 = сильный урок)

            if not prompt or not completion:
                self._send_error(400, "Нужны prompt и completion")
                return

            # Формируем полный текст для обучения: user + model
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
            train_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            
            # Токенизация с labels (модель учится предсказывать completion)
            enc = tokenizer(
                train_prompt, return_tensors="pt", truncation=True, max_length=2048,
                add_special_tokens=False
            ).to(model.device)
            
            input_ids = enc["input_ids"]
            # Находим, где начинается ответ ассистента, чтобы считать loss только по нему
            user_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            user_len = len(tokenizer(user_prompt, add_special_tokens=False)["input_ids"])
            
            labels = input_ids.clone()
            labels[:, :user_len] = -100 # Игнорируем loss на промпте пользователя

            # ШАГ ОБУЧЕНИЯ
            model.train() # Включаем режим обучения (включает dropout для LoRA)
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss * reward # Учитываем "ценность" урока
            loss.backward()
            
            # Градиентный клиппинг для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.eval() # Возвращаем в режим инференса

            print(f"  🧠 ОБУЧЕНИЕ: loss={loss.item():.4f}, reward={reward}", flush=True)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'trained', 'loss': float(loss.item())}).encode('utf-8'))

        except Exception as e:
            print(f"  ❌ Ошибка обучения: {e}", flush=True)
            traceback.print_exc()
            self._send_error(500, str(e))

    def _send_error(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({'error': message}).encode())

    def log_message(self, format, *args):
        if args and 'OPTIONS' not in str(args[0]):
            print(f"  🌐 {args[0]}", flush=True)

# ... (остальной код запуска server.serve_forever() остается без изменений)

class ModelHandler(BaseHTTPRequestHandler):

    def _set_cors_headers(self):
        """Устанавливает CORS-заголовки для всех ответов."""
        self.send_header('Access-Control-Allow-Origin', ALLOWED_ORIGINS)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '86400')  # кэш preflight на 24 часа

    def do_OPTIONS(self):
        """Обработка preflight-запросов от браузера."""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self._set_cors_headers()  # ← ДОБАВЛЕНО
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode())

    def do_POST(self):
        try:
            content_length = self.headers.get('Content-Length')
            if content_length is None:
                self._send_error(400, "Нет Content-Length")
                return
            length = int(content_length)
            if length == 0:
                self._send_error(400, "Пустое тело")
                return

            raw_body = self.rfile.read(length)
            body = json.loads(raw_body.decode('utf-8'))

            prompt_from_client = body.get('prompt', '')
            max_tokens = body.get('max_tokens', 80)

            if not prompt_from_client.strip():
                self._send_error(400, "Пустой prompt")
                return

            messages = [{"role": "user", "content": prompt_from_client}]
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            inputs = tokenizer(
                chat_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                add_special_tokens=False,
            ).to(model.device)

            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    min_new_tokens=1,
                    do_sample=False,
                    repetition_penalty=1.1,
                    eos_token_id=stop_ids,
                    bad_words_ids=[[i] for i in bad_ids],
                )

            generated_ids = outputs[0][input_length:]
            response_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True,
            ).strip()

            for tok in ('<end_of_turn>', '<eos>', '<bos>', '<pad>',
                        '<start_of_turn>', 'model\n'):
                response_text = response_text.replace(tok, '')
            response_text = response_text.strip()

            print(f"  📥 {len(raw_body)}b → gen={generated_ids.tolist()} "
                  f"→ {repr(response_text[:80])}", flush=True)

            if not response_text:
                response_text = "ДЕЙСТВИЕ: ФОЛД | СУММА: 0"

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._set_cors_headers()  # ← ДОБАВЛЕНО
            self.end_headers()
            self.wfile.write(json.dumps(
                {'response': response_text}, ensure_ascii=False,
            ).encode('utf-8'))

        except Exception as e:
            print(f"  ❌ Ошибка: {e}", flush=True)
            traceback.print_exc()
            try:
                self._send_error(500, str(e))
            except Exception:
                pass

    def do_PUT(self):
        """Обучение: сохранить руку в память и, возможно, обучить батч"""
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length).decode('utf-8'))
            
            # 1. Сохраняем в постоянную память (для In-Context Learning)
            save_memory({
                "stage": body.get("stage", "unknown"),
                "prompt_summary": body.get("prompt", "")[:100] + "...",
                "completion": body.get("completion", ""),
                "result": body.get("result", "unknown"),
                "reward": body.get("reward", 1.0)
            })
            
            # 2. Добавляем в буфер для LoRA-обучения
            training_buffer.append({
                "prompt": body.get("prompt", ""),
                "completion": body.get("completion", ""),
                "reward": body.get("reward", 1.0)
            })
            
            # 3. Проверяем, пора ли обучать
            train_status = train_batch()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._set_cors()
            self.end_headers()
            self.wfile.write(json.dumps(train_status, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            self._send_error(500, str(e))

# ============================================================
# УПРАВЛЕНИЕ ПАМЯТЬЮ И ОБУЧЕНИЕМ
# ============================================================

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_memory(hand_data):
    memory = load_memory()
    memory.append(hand_data)
    if len(memory) > 100: # Храним только последние 100 раздач для промпта
        memory = memory[-100:]
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def train_batch():
    global last_loss, training_buffer
    if not optimizer or len(training_buffer) < MEMORY_LIMIT:
        return {"status": "waiting", "buffer_size": len(training_buffer)}
    
    print(f"\n🧠 НАЧИНАЮ ОБУЧЕНИЕ НА БАТЧЕ ИЗ {len(training_buffer)} ПРИМЕРОВ...")
    model.train()
    total_loss = 0.0
    
    for item in training_buffer:
        prompt = item['prompt']
        completion = item['completion']
        reward = item.get('reward', 1.0)
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        train_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(train_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        
        user_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        user_len = len(tokenizer(user_text, add_special_tokens=False)["input_ids"])
        
        labels = enc["input_ids"].clone()
        labels[:, :user_len] = -100
        
        optimizer.zero_grad()
        outputs = model(input_ids=enc["input_ids"], labels=labels)
        loss = outputs.loss * reward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    last_loss = total_loss / len(training_buffer)
    model.eval()
    
    # СОХРАНЕНИЕ ВЕСОВ ПОСЛЕ ОБУЧЕНИЯ!
    model.save_pretrained(LORA_PATH)
    tokenizer.save_pretrained(LORA_PATH)
    print(f"✅ ОБУЧЕНИЕ ЗАВЕРШЕНО. Loss: {last_loss:.4f}. Веса LoRA сохранены в {LORA_PATH}")
    
    training_buffer = [] # Очистка буфера
    return {"status": "trained", "loss": last_loss, "buffer_size": 0}

server = HTTPServer((HOST, PORT), ModelHandler)
print(f"\n🚀 Сервер: http://{HOST}:{PORT}")
print("Ctrl+C для остановки.\n")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Остановлен")
    server.server_close()