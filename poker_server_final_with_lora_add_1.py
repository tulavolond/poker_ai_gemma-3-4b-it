"""
Сервер покерного ИИ с памятью и обучением LoRA.
Запуск: python model_server.py
"""
import os
import json
import traceback
import torch
import threading
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from http.server import HTTPServer, BaseHTTPRequestHandler

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

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

# ============================================================
# КВАНТИЗАЦИЯ (как в оригинале)
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

device_map = {"": 0} if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map=device_map,
    dtype=torch.bfloat16,
    token=HF_TOKEN,
    low_cpu_mem_usage=True,
)

# ============================================================
# ТОКЕНИЗАТОР (как в оригинале)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

USE_LORA = True
if USE_LORA and os.path.exists(LORA_PATH):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, LORA_PATH)
    print("✅ LoRA подключен из", LORA_PATH)
else:
    print("⚠️ LoRA отключен или путь не найден.")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

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

# ============================================================
# СТОП-ТОКЕНЫ И BAD WORDS (КРИТИЧЕСКИ ВАЖНО!)
# ============================================================
end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
print("end_of_turn id:", end_of_turn_id)

stop_ids = []
if tokenizer.eos_token_id is not None:
    stop_ids.append(tokenizer.eos_token_id)
if (end_of_turn_id is not None 
    and end_of_turn_id != tokenizer.unk_token_id 
    and end_of_turn_id != tokenizer.eos_token_id):
    stop_ids.append(end_of_turn_id)
stop_ids = sorted(set(stop_ids))
print("STOP IDs:", stop_ids)

bad_ids = []
for tid in (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id):
    if tid is not None:
        bad_ids.append(tid)
bad_ids = sorted(set(bad_ids))
print("BAD IDs:", bad_ids)

# ============================================================
# ОПТИМИЗАТОР ДЛЯ ОБУЧЕНИЯ (если LoRA есть)
# ============================================================
if USE_LORA and os.path.exists(LORA_PATH):
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
    print("✅ Оптимизатор для обучения готов")
else:
    optimizer = None
    print("⚠️ Оптимизатор не создан")

print("✅ Модель готова")

# ============================================================
# ПАМЯТЬ И БУФЕР ОБУЧЕНИЯ
# ============================================================
training_buffer = []
MEMORY_LIMIT = 30  # Обучаем батчами по 30 раздач
last_loss = 0.0
memory_lock = threading.Lock()

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_memory(hand_data):
    with memory_lock:
        memory = load_memory()
        memory.append(hand_data)
        if len(memory) > 100:
            memory = memory[-100:]
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)

def train_batch():
    global last_loss, training_buffer
    if not optimizer or len(training_buffer) < MEMORY_LIMIT:
        return {"status": "waiting", "buffer_size": len(training_buffer)}
    
    print(f"\n🧠 ОБУЧЕНИЕ НА БАТЧЕ {len(training_buffer)} рук...")
    model.train()
    total_loss = 0.0
    
    for item in training_buffer:
        messages = [
            {"role": "user", "content": item['prompt']},
            {"role": "assistant", "content": item['completion']}
        ]
        train_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(train_text, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
        
        user_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": item['prompt']}], 
            tokenize=False, add_generation_prompt=True
        )
        user_len = len(tokenizer(user_text, add_special_tokens=False)["input_ids"])
        
        labels = enc["input_ids"].clone()
        labels[:, :user_len] = -100
        
        optimizer.zero_grad()
        outputs = model(input_ids=enc["input_ids"], labels=labels)
        loss = outputs.loss * item.get('reward', 1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    last_loss = total_loss / len(training_buffer)
    model.eval()
    
    # СОХРАНЕНИЕ ВЕСОВ НА ДИСК
    model.save_pretrained(LORA_PATH)
    tokenizer.save_pretrained(LORA_PATH)
    print(f"✅ Обучено. Loss: {last_loss:.4f}. Веса сохранены.")
    
    training_buffer = []
    return {"status": "trained", "loss": last_loss, "buffer_size": 0}

# ============================================================
# ДИАГНОСТИКА ПРИ СТАРТЕ
# ============================================================
def _quick_test():
    print("\n" + "=" * 60)
    print("  ДИАГНОСТИКА МОДЕЛИ")
    print("=" * 60)
    
    messages = [{"role": "user", "content": "Скажи одно слово: привет"}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            eos_token_id=stop_ids,
            bad_words_ids=[[i] for i in bad_ids],
        )
    
    gen = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    print(f"  Тестовый ответ: {repr(text)}")
    
    if not text:
        print("  ❌ МОДЕЛЬ НЕ ГЕНЕРИРУЕТ ТЕКСТ!")
    else:
        print("  ✅ Модель работает\n")
    print("=" * 60 + "\n")

_quick_test()

# ============================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ИНФЕРЕНСА
# ============================================================
def run_inference(prompt_from_client, max_tokens=120):
    """Запуск инференса с правильными параметрами."""
    messages = [{"role": "user", "content": prompt_from_client}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        chat_prompt, return_tensors="pt",
        truncation=True, max_length=2048,
        add_special_tokens=False,
    ).to(model.device)
    
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_new_tokens=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=stop_ids,
            bad_words_ids=[[i] for i in bad_ids],
        )
    
    generated_ids = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Очистка от мусора
    for tok in ('<end_of_turn>', '<eos>', '<bos>', '<pad>', '<start_of_turn>', 'model\n', 'user\n'):
        response_text = response_text.replace(tok, '')
    response_text = response_text.strip()
    
    return response_text, generated_ids.tolist()

# ============================================================
# HTTP-ОБРАБОТЧИК С CORS
# ============================================================
ALLOWED_ORIGINS = "*"

class ModelHandler(BaseHTTPRequestHandler):
    def _set_cors(self):
        self.send_header('Access-Control-Allow-Origin', ALLOWED_ORIGINS)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        """Возвращает статус и память"""
        try:
            memory = load_memory()[-5:]
            response = {
                'status': 'ok',
                'lora_active': USE_LORA,
                'buffer_size': len(training_buffer),
                'last_loss': last_loss,
                'recent_memory': memory
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            self._send_error(500, str(e))

    # def do_POST(self):
    #     """Инференс: получить решение ИИ"""
    #     try:
    #         content_length = self.headers.get('Content-Length')
    #         if content_length is None:
    #             self._send_error(400, "Нет Content-Length")
    #             return
    #         length = int(content_length)
    #         if length == 0:
    #             self._send_error(400, "Пустое тело")
    #             return

    #         raw_body = self.rfile.read(length)
    #         body = json.loads(raw_body.decode('utf-8'))
    #         prompt_from_client = body.get('prompt', '')
    #         max_tokens = body.get('max_tokens', 120)

    #         if not prompt_from_client.strip():
    #             self._send_error(400, "Пустой prompt")
    #             return

    #         print(f"\n📥 POST запрос ({len(raw_body)} байт)", flush=True)
    #         print(f"   Промпт: {prompt_from_client[:150]}...", flush=True)
            
    #         response_text, gen_ids = run_inference(prompt_from_client, max_tokens)
            
    #         print(f"   Сгенерированные ID: {gen_ids}", flush=True)
    #         print(f"   Ответ: {repr(response_text)}", flush=True)
            
    #         # Fallback если ответ пустой
    #         if not response_text:
    #             response_text = "АНАЛИЗ: Нет информации\nДЕЙСТВИЕ: FOLD\nСУММА: 0"
    #             print("   ⚠️ Пустой ответ, использую fallback", flush=True)

    #         self.send_response(200)
    #         self.send_header('Content-Type', 'application/json; charset=utf-8')
    #         self._set_cors()
    #         self.end_headers()
    #         self.wfile.write(json.dumps(
    #             {'response': response_text}, ensure_ascii=False,
    #         ).encode('utf-8'))

    #     except Exception as e:
    #         print(f"❌ Ошибка POST: {e}", flush=True)
    #         traceback.print_exc()
    #         self._send_error(500, str(e))

    def do_POST(self):
        """Инференс: получить решение ИИ"""
        try:
            content_length = self.headers.get('Content-Length')
            if content_length is None:
                self._send_error(400, "Нет Content-Length")
                return
            length = int(content_length)
            if length == 0:
                self._send_error(400, "Пустое тело (length=0)")
                return

            raw_body = self.rfile.read(length)
            print(f"\n🔍 DEBUG: Получено байт: {length}")
            print(f"🔍 DEBUG: Сырые данные: {raw_body}") # <-- ЭТО ПОКАЖЕТ, ЧТО ПРИШЛО
            
            try:
                body = json.loads(raw_body.decode('utf-8'))
            except json.JSONDecodeError as e:
                print(f"❌ DEBUG: Ошибка парсинга JSON. Текст: {raw_body.decode('utf-8', errors='ignore')}")
                self._send_error(400, f"Неверный JSON: {str(e)}")
                return

            prompt_from_client = body.get('prompt', '')
            max_tokens = body.get('max_tokens', 120)

            if not prompt_from_client.strip():
                self._send_error(400, "Пустой prompt внутри JSON")
                return

            print(f"📥 Запрос на генерацию (длина промпта: {len(prompt_from_client)})", flush=True)
            response_text, gen_ids = run_inference(prompt_from_client, max_tokens)
            
            if not response_text:
                response_text = "АНАЛИЗ: Нет информации\nДЕЙСТВИЕ: FOLD\nСУММА: 0"

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            self.wfile.write(json.dumps({'response': response_text}, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            print(f"❌ Критическая ошибка POST: {e}", flush=True)
            traceback.print_exc()
            self._send_error(500, str(e))

    def do_PUT(self):
        """Обучение: сохранить руку"""
        if not optimizer or not (USE_LORA and os.path.exists(LORA_PATH)):
            self._send_error(501, "Обучение не настроено")
            return
        
        try:
            length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(length)
            body = json.loads(raw_body.decode('utf-8'))
            
            prompt = body.get('prompt', '')
            completion = body.get('completion', '')
            reward = body.get('reward', 1.0)
            
            if not prompt or not completion:
                self._send_error(400, "Нужны prompt и completion")
                return
            
            # Сохраняем в память
            save_memory({
                "stage": body.get("stage", "?"),
                "result": body.get("result", "?"),
                "completion": completion[:200]
            })
            
            # В буфер обучения
            with memory_lock:
                training_buffer.append({
                    "prompt": prompt,
                    "completion": completion,
                    "reward": reward
                })
            
            # Пробуем обучить
            train_status = train_batch()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            self.wfile.write(json.dumps(train_status, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            print(f"❌ Ошибка PUT: {e}", flush=True)
            traceback.print_exc()
            self._send_error(500, str(e))

    def _send_error(self, code, message):
        try:
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            self.wfile.write(json.dumps({'error': message}, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            print(f"Не удалось отправить ошибку: {e}")

    def log_message(self, format, *args):
        # Только не-OPTIONS для чистоты
        if args and 'OPTIONS' not in str(args[0]):
            print(f"🌐 {args[0]}", flush=True)

# ============================================================
# ЗАПУСК СЕРВЕРА
# ============================================================
server = HTTPServer((HOST, PORT), ModelHandler)
print(f"\n🚀 Сервер: http://{HOST}:{PORT}")
print(f"🧠 Память: {MEMORY_FILE}")
print("Ctrl+C для остановки.\n")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Останавливаю...")
    if USE_LORA and optimizer:
        print("💾 Сохраняю финальные веса LoRA...")
        model.save_pretrained(LORA_PATH)
        tokenizer.save_pretrained(LORA_PATH)
    server.server_close()