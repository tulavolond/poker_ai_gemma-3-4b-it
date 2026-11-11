"""
server_with_trained_model.py - Покерный сервер с обученной моделью для префлопа.
ИИ_1 использует обученную LoRA модель для префлоп-решений.

Запуск: python server_with_trained_model.py
"""
import os
import json
import traceback
import threading
import torch
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler

# Импортируем анализатор
from analyzer import analyze_preflop, build_deck, Card

load_dotenv()

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PREFLOP_PATH = "./lora_preflop_stage1"
PORT = 8899
HOST = "192.168.1.103"

# Файлы для памяти и статистики
MEMORY_FILE = "./poker_memory.json"
BUFFER_FILE = "./training_buffer.json"
STATS_FILE = "./training_stats.json"

print("=" * 60)
print("  ЗАГРУЗКА МОДЕЛЕЙ")
print("=" * 60)

# ============================================================
# 1. ЗАГРУЗКА БАЗОВОЙ МОДЕЛИ
# ============================================================
print("📦 Загрузка базовой модели...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("✅ Базовая модель загружена")

# ============================================================
# 2. ЗАГРУЗКА ОБУЧЕННОЙ LoRA ДЛЯ ПРЕФЛОПА
# ============================================================
if os.path.exists(LORA_PREFLOP_PATH):
    print(f"🎯 Загрузка обученной LoRA для префлопа из {LORA_PREFLOP_PATH}...")
    preflop_model = PeftModel.from_pretrained(base_model, LORA_PREFLOP_PATH)
    preflop_model.eval()
    print("✅ Обученная префлоп-модель загружена")
    PREFLOP_MODEL_AVAILABLE = True
else:
    print(f"⚠️ Обученная LoRA не найдена в {LORA_PREFLOP_PATH}")
    print("   Используем базовую модель для всех решений")
    preflop_model = base_model
    PREFLOP_MODEL_AVAILABLE = False

print()

# ============================================================
# 3. ПАМЯТЬ И СТАТИСТИКА
# ============================================================
training_buffer = []
memory_cache = []
stats = {"pretrain_examples": 0, "online_examples": 0, "batches_trained": 0}
save_lock = threading.Lock()


def load_memory():
    global memory_cache
    if memory_cache:
        return memory_cache
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                memory_cache = json.load(f)
                return memory_cache
        except:
            memory_cache = []
    return []


def add_to_memory(hand_data):
    global memory_cache
    memory_cache = load_memory()
    memory_cache.append(hand_data)
    if len(memory_cache) > 100:
        memory_cache = memory_cache[-100:]
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Ошибка записи памяти: {e}")


# ============================================================
# 4. ИНФЕРЕНС С ОБУЧЕННОЙ МОДЕЛЬЮ
# ============================================================
def generate_response(model, prompt, max_new_tokens=100, temperature=0.7):
    """Генерирует ответ модели."""
    messages = [{"role": "user", "content": prompt}]
    
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    response = response.strip()
    for tok in ('<|im_end|>', '<|im_start|>', 'assistant', 'user'):
        response = response.replace(tok, '')
    
    return response.strip()


def parse_decision(response_text):
    """Извлекает решение из ответа модели."""
    import re
    
    action_match = re.search(r'ДЕЙСТВИЕ:\s*(FOLD|CALL|RAISE|CHECK)', response_text, re.IGNORECASE)
    amount_match = re.search(r'СУММА:\s*(\d+)', response_text)
    analysis_match = re.search(r'АНАЛИЗ:\s*(.+?)(?=\n|$)', response_text)
    
    action = action_match.group(1).upper() if action_match else "FOLD"
    amount = int(amount_match.group(1)) if amount_match else 0
    analysis = analysis_match.group(1).strip() if analysis_match else ""
    
    return {
        "action": action,
        "amount": amount,
        "analysis": analysis
    }


def run_preflop_inference(prompt):
    """Инференс для префлопа с обученной моделью."""
    if PREFLOP_MODEL_AVAILABLE:
        print("🎯 [ПРЕФЛОП] Используем ОБУЧЕННУЮ модель")
        response = generate_response(preflop_model, prompt, temperature=0.3)
    else:
        print("⚠️ [ПРЕФЛОП] Используем базовую модель (обученная недоступна)")
        response = generate_response(base_model, prompt, temperature=0.7)
    
    return response


def run_postflop_inference(prompt):
    """Инференс для постфлопа (флоп/тёрн/ривер) с базовой моделью."""
    print("📊 [ПОСТФЛОП] Используем базовую модель")
    response = generate_response(base_model, prompt, temperature=0.7)
    return response


# ============================================================
# 5. HTTP-ОБРАБОТЧИК
# ============================================================
class ModelHandler(BaseHTTPRequestHandler):
    timeout = 120

    def _set_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _safe_write(self, data):
        try:
            self.wfile.write(data)
            return True
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return False

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        try:
            if self.path == "/stats":
                memory = load_memory()[-50:]
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self._set_cors()
                self.end_headers()
                data = json.dumps({
                    'status': 'ok',
                    'preflop_model_available': PREFLOP_MODEL_AVAILABLE,
                    'stats': stats,
                    'buffer_size': len(training_buffer),
                    'recent_memory': memory
                }, ensure_ascii=False).encode('utf-8')
                self._safe_write(data)
                return

            # Обычный GET
            self.send_response(200)
            self._set_cors()
            self.end_headers()
            self._safe_write(b"Server is running")

        except Exception as e:
            print(f"❌ Ошибка GET: {e}")
            self._send_error(500, str(e))

    def do_POST(self):
        """Обработка запросов от ИИ."""
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._send_error(400, "Empty body")
                return

            raw_body = self.rfile.read(length)
            body = json.loads(raw_body.decode('utf-8'))

            prompt = body.get('prompt', '')
            stage = body.get('stage', 'unknown')
            
            if not prompt:
                self._send_error(400, "Empty prompt")
                return

            print(f"\n📥 Запрос от ИИ. Стадия: {stage}")

            # Выбираем модель в зависимости от стадии
            if stage.lower() == 'preflop':
                response_text = run_preflop_inference(prompt)
            else:
                response_text = run_postflop_inference(prompt)

            print(f"📤 Ответ модели (первые 300 символов):\n{response_text[:300]}")

            # Парсим решение
            decision = parse_decision(response_text)
            print(f"✅ Решение: {decision['action']} {decision['amount']}")
            print(f"   Анализ: {decision['analysis']}")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            
            data = json.dumps({
                'response': response_text,
                'decision': decision
            }, ensure_ascii=False).encode('utf-8')
            
            if not self._safe_write(data):
                print("⚠️ Клиент разорвал соединение")

        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            print("⚠️ Клиент разорвал соединение")
        except Exception as e:
            print(f"❌ Ошибка POST: {e}")
            traceback.print_exc()
            self._send_error(500, str(e))

    def do_PUT(self):
        """Сохранение результатов игры."""
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._send_error(400, "Empty body")
                return
            
            raw_body = self.rfile.read(length)
            body = json.loads(raw_body.decode('utf-8'))

            hand_data = {
                "stage": body.get("stage", "?"),
                "my_cards": body.get("my_cards", "?"),
                "board": body.get("board", "?"),
                "pot": body.get("pot", 0),
                "to_call": body.get("to_call", 0),
                "stack": body.get("stack", 0),
                "action_taken": body.get("action_taken", "?"),
                "outcome": body.get("outcome", "?"),
                "lesson": body.get("lesson", "")
            }
            add_to_memory(hand_data)
            stats["online_examples"] += 1

            print(f"📝 Сохранена рука: {hand_data['stage']}, действие: {hand_data['action_taken']}, результат: {hand_data['outcome']}")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            self._safe_write(json.dumps({"status": "saved"}, ensure_ascii=False).encode('utf-8'))

        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            print("⚠️ PUT: клиент разорвал соединение")
        except Exception as e:
            print(f"❌ Ошибка PUT: {e}")
            self._send_error(500, str(e))

    def _send_error(self, code, message):
        try:
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            self._safe_write(json.dumps({'error': message}, ensure_ascii=False).encode('utf-8'))
        except Exception:
            pass

    def log_message(self, format, *args):
        if 'OPTIONS' not in str(args[0]):
            print(f"🌐 {args[0]}")


# ============================================================
# 6. ЗАПУСК СЕРВЕРА
# ============================================================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


server = ThreadedHTTPServer((HOST, PORT), ModelHandler)

print("=" * 60)
print("🚀 СЕРВЕР ЗАПУЩЕН")
print("=" * 60)
print(f"📡 Адрес: http://{HOST}:{PORT}")
print(f"🎯 Префлоп-модель: {'✅ Обученная' if PREFLOP_MODEL_AVAILABLE else '⚠️ Базовая'}")
print(f"📊 Статистика: http://{HOST}:{PORT}/stats")
print()
print("ИИ_1 теперь использует обученную модель для префлоп-решений!")
print("Нажмите Ctrl+C для остановки.")
print()

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Остановка сервера...")
    server.server_close()
    print("✅ Сервер остановлен.")