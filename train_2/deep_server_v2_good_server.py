"""
poker_server_v4.py - Покерный сервер с LoRA + постобработкой.
Работает даже с недообученной LoRA благодаря fix_response.

Запуск: python poker_server_v4.py
"""
import os
import re
import json
import time
import threading
import torch
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler
from peft import PeftModel

load_dotenv()

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./train_2/dataset_2000_v2"
LORA_ONLINE_PATH = "./train_2/lora_online_learning"

MEMORY_FILE = "./train_2/poker_memory.json"
BUFFER_FILE = "./train_2/training_buffer.json"
STATS_FILE = "./train_2/training_stats.json"
HISTORY_FILE = "./train_2/chat_memory.json"

PORT = 8899
HOST = "0.0.0.0"
MAX_MESSAGES = 40
MEMORY_LIMIT = 100
MAX_SEQ_LEN = 788   # ← как в обучении

# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================
print("=" * 60)
print("  ЗАГРУЗКА МОДЕЛИ")
print("=" * 60)

print(f"📦 Загрузка модели: {BASE_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID, trust_remote_code=True, padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"✅ Модель загружена на: {model.device}")
print(f"   Память GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

LORA_AVAILABLE = False
if os.path.exists(LORA_PATH):
    try:
        print(f"\n🎯 Загрузка LoRA из {LORA_PATH}...")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model.eval()
        LORA_AVAILABLE = True
        print("✅ LoRA загружена")
    except Exception as e:
        print(f"⚠️ Ошибка LoRA: {e}")
else:
    print(f"\n⚠️ LoRA не найдена: {LORA_PATH}")

# ============================================================
# ПАМЯТЬ / БУФЕР / СТАТИСТИКА
# ============================================================
training_buffer = []
memory_cache = []
last_loss = 0.0
training_lock = threading.Lock()
save_lock = threading.Lock()
save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Saver")

stats = {
    "pretrain_examples": 0,
    "online_examples": 0,
    "batches_trained": 0,
    "preflop_queries": 0,
    "postflop_queries": 0,
}


def load_all_data():
    global training_buffer, memory_cache, stats
    for path, target in [(MEMORY_FILE, "memory"), (BUFFER_FILE, "buffer"), (STATS_FILE, "stats")]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if target == "memory":
                    memory_cache = data
                    print(f"📂 Память: {len(memory_cache)}")
                elif target == "buffer":
                    training_buffer = data
                    print(f"📂 Буфер: {len(training_buffer)}")
                elif target == "stats":
                    stats.update(data)
                    print(f"📂 Статистика загружена")
            except Exception:
                pass


def save_all_data():
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory_cache[-100:], f, ensure_ascii=False, indent=1)
        with open(BUFFER_FILE, 'w', encoding='utf-8') as f:
            json.dump(training_buffer, f, ensure_ascii=False, indent=1)
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Ошибка сохранения: {e}")


def add_to_memory(hand_data):
    global memory_cache
    memory_cache.append(hand_data)
    if len(memory_cache) > 100:
        memory_cache = memory_cache[-100:]
    save_all_data()


# ============================================================
# ИСТОРИЯ (без неё — модель забывает контекст)
# ============================================================
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def save_history(messages):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def trim_history(messages, max_messages=MAX_MESSAGES):
    if len(messages) <= max_messages:
        return messages
    trimmed = messages[-max_messages:]
    if trimmed and trimmed[0]["role"] == "assistant":
        trimmed = trimmed[1:]
    return trimmed


# ============================================================
# ДЕТЕКТОРЫ
# ============================================================
def detect_stage(prompt_text):
    p = prompt_text.upper()
    if "ПРЕФЛОП" in p or "PRE-FLOP" in p:
        return "preflop"
    if "ФЛОП" in p or "FLOP" in p:
        return "flop"
    if "ТЁРН" in p or "TURN" in p:
        return "turn"
    if "РИВЕР" in p or "RIVER" in p:
        return "river"
    return "unknown"


# ============================================================
# ИНФЕРЕНС — БЕЗ STAGE_INSTRUCTIONS, БЕЗ FEW_SHOT
# ============================================================
# def run_inference(prompt_text: str, max_tokens=150, stage_hint=None):
#     """
#     [КРИТИЧНО] НЕ добавляем STAGE_INSTRUCTIONS и FEW_SHOT!
#     Модель училась на чистом промпте из dataset_v2.py.
#     """
#     stage = stage_hint or detect_stage(prompt_text)
#     if stage not in ("preflop", "flop", "turn", "river"):
#         stage = "flop"

#     # Если промпт не начинается с [СТАДИЯ: — добавляем (на всякий случай)
#     if not prompt_text.strip().startswith("[СТАДИЯ:"):
#         prompt_text = f"[СТАДИЯ: {stage.upper()}]\n\n{prompt_text}"

#     # [КРИТИЧНО] НЕ добавляем ничего!
#     full_prompt = prompt_text

#     # История (для контекста, но не для инструкций)
#     messages = load_history()
#     messages.append({"role": "user", "content": full_prompt})
#     messages = trim_history(messages)

#     chat_prompt = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = tokenizer(
#         chat_prompt, return_tensors="pt",
#         truncation=True, max_length=MAX_SEQ_LEN
#     ).to(model.device)
#     input_length = inputs["input_ids"].shape[1]

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             do_sample=False,          # [КРИТИЧНО] детерминированный ответ
#             pad_token_id=tokenizer.pad_token_id,
#         )

#     gen_ids = outputs[0][input_length:]
#     text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
#     for tok in ('<|im_end|>', '<|im_start|>', 'assistant', 'user', '<end_of_turn>'):
#         text = text.replace(tok, '').strip()

#     # Валидация формата
#     if 'ДЕЙСТВИЕ:' not in text:
#         text = "АНАЛИЗ: Не уверен, играю осторожно.\nДЕЙСТВИЕ: FOLD\nСУММА: 0"
#     elif 'СУММА:' not in text:
#         text += "\nСУММА: 0"

#     messages.append({"role": "assistant", "content": text})
#     save_history(trim_history(messages))

#     return text

def run_inference(prompt_text: str, max_tokens=150, stage_hint=None):
    stage = stage_hint or detect_stage(prompt_text)
    if stage not in ("preflop", "flop", "turn", "river"):
        stage = "flop"

    if not prompt_text.strip().startswith("[СТАДИЯ:"):
        prompt_text = f"[СТАДИЯ: {stage.upper()}]\n\n{prompt_text}"

    # [КРИТИЧНО] НЕ используем chat_template — модель училась без него!
    # Просто токенизируем промпт как есть
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = outputs[0][input_length:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    for tok in ('<|im_end|>', '<|im_start|>', 'assistant', 'user', '<end_of_turn>'):
        text = text.replace(tok, '').strip()

    if 'ДЕЙСТВИЕ:' not in text:
        text = "АНАЛИЗ: Не уверен.\nДЕЙСТВИЕ: FOLD\nСУММА: 0"
    elif 'СУММА:' not in text:
        text += "\nСУММА: 0"

    return text

# ============================================================
# ОБУЧЕНИЕ (online)
# ============================================================
def train_batch():
    global last_loss, training_buffer, stats

    with training_lock:
        if len(training_buffer) < MEMORY_LIMIT:
            return {"status": "waiting", "buffer_size": len(training_buffer)}
        batch = training_buffer[:]
        training_buffer = []
        save_all_data()

    print(f"\n🧠 ОБУЧЕНИЕ: {len(batch)} примеров...")
    model.train()
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5, weight_decay=0.01
    )
    optimizer.zero_grad()

    total_loss = 0.0
    valid_count = 0

    for i, item in enumerate(batch):
        try:
            messages = [
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": item['completion']}
            ]
            train_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            enc = tokenizer(
                train_text, return_tensors="pt",
                truncation=True, max_length=MAX_SEQ_LEN
            ).to(model.device)

            user_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": item['prompt']}],
                tokenize=False, add_generation_prompt=True
            )
            user_len = len(tokenizer(user_text, add_special_tokens=False)["input_ids"])

            labels = enc["input_ids"].clone()
            labels[:, :user_len] = -100

            outputs = model(input_ids=enc["input_ids"], labels=labels)
            reward = item.get('reward', 1.0)
            loss = outputs.loss * reward
            loss.backward()

            total_loss += loss.item()
            valid_count += 1
        except Exception as e:
            print(f"⚠️ Пропуск {i}: {e}")
            continue

    if valid_count > 0:
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )
        optimizer.step()
        last_loss = total_loss / valid_count
        stats["batches_trained"] += 1
        save_executor.submit(save_lora_weights)
        print(f"✅ Loss: {last_loss:.4f}")
    else:
        last_loss = 0.0

    model.eval()
    save_all_data()
    return {"status": "trained", "loss": last_loss, "buffer_size": 0}


def save_lora_weights():
    try:
        with save_lock:
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(LORA_ONLINE_PATH)
                tokenizer.save_pretrained(LORA_ONLINE_PATH)
            print(f"✅ LoRA сохранены в {LORA_ONLINE_PATH}")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения: {e}")


# ============================================================
# HTTP-ОБРАБОТЧИК
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
            parsed = urlparse(self.path)

            if parsed.path in ("/stats", "/", "/health"):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self._set_cors()
                self.end_headers()
                self._safe_write(json.dumps({
                    'status': 'ok',
                    'lora_available': LORA_AVAILABLE,
                    'buffer_size': len(training_buffer),
                    'last_loss': last_loss,
                    'device': str(model.device),
                    'stats': stats,
                }, ensure_ascii=False).encode('utf-8'))
                return

            self._send_error(404, "Not found")

        except Exception as e:
            self._send_error(500, str(e))

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._send_error(400, "Empty body")
                return

            raw_body = self.rfile.read(length)
            body = json.loads(raw_body.decode('utf-8'))
            prompt = body.get('prompt', '')
            if not prompt:
                self._send_error(400, "Empty prompt")
                return

            start = time.time()
            stage = detect_stage(prompt)

            # [ОТЛАДКА] — видно, что реально приходит с сайта
            print("=" * 60)
            print(f"📥 POST stage={stage}, prompt={len(prompt)} симв.")
            print(f"[PROMPT] {prompt[:500]}")
            print("=" * 60)

            response_text = run_inference(prompt, body.get('max_tokens', 150))

            elapsed = time.time() - start

            if stage == "preflop":
                stats["preflop_queries"] += 1
            else:
                stats["postflop_queries"] += 1

            print(f"📤 Ответ за {elapsed:.1f}s:")
            print(f"{response_text}")
            print("=" * 60)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            data = json.dumps({
                'response': f"📤 Ответ за {elapsed:.1f}s\n" + response_text
            }, ensure_ascii=False).encode('utf-8')
            if not self._safe_write(data):
                print("⚠️ Клиент разорвал соединение")

        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            print("⚠️ Клиент разорвал")
        except Exception as e:
            print(f"❌ Ошибка POST: {e}")
            self._send_error(500, str(e))

    def do_PUT(self):
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

            stage = body.get("stage", "flop").lower()
            action = body.get("action_taken", "FOLD")
            outcome = body.get("outcome", "LOSS")
            lesson = body.get("lesson", "")

            prompt = f"""СТАДИЯ: {stage.upper()}
ТВОИ КАРТЫ: {hand_data['my_cards']}
БОРД: {hand_data['board']}
БАНК: ${hand_data['pot']}, СТЕК: ${hand_data['stack']}
НУЖНО КОЛЛИРОВАТЬ: ${hand_data['to_call']}

Ответь СТРОГО:
АНАЛИЗ: [короткое объяснение]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/RAISE]
СУММА: [число]"""

            parts = str(action).split()
            act_word = parts[0].upper() if parts else 'FOLD'
            amount = parts[1].replace('$', '') if len(parts) > 1 else '0'

            completion = f"""АНАЛИЗ: {lesson}
ДЕЙСТВИЕ: {act_word}
СУММА: {amount}"""

            reward = 1.0 if 'WIN' in outcome else 0.2

            training_buffer.append({
                "prompt": prompt, "completion": completion,
                "reward": reward, "stage": stage
            })

            save_all_data()
            print(f"📝 В буфере: {len(training_buffer)}/{MEMORY_LIMIT}")

            train_status = train_batch()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            self._safe_write(json.dumps(train_status, ensure_ascii=False).encode('utf-8'))

        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            print("⚠️ PUT: клиент разорвал")
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
# ЗАПУСК
# ============================================================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


print("\n📂 Загрузка сохранённых данных...")
load_all_data()

server = ThreadedHTTPServer((HOST, PORT), ModelHandler)

print()
print("=" * 60)
print("🚀 СЕРВЕР ЗАПУЩЕН")
print("=" * 60)
print(f"📡 Адрес: http://{HOST}:{PORT}")
print(f"🎯 LoRA: {'✅ Обученная' if LORA_AVAILABLE else '⚠️ Базовая'}")
print(f"📏 MAX_SEQ_LEN: {MAX_SEQ_LEN}")
print(f"🔧 Без STAGE_INSTRUCTIONS и FEW_SHOT — чистый промпт")
print(f"📊 Статистика: http://{HOST}:{PORT}/stats")
print()

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Остановка...")
    save_all_data()
    save_lora_weights()
    save_executor.shutdown(wait=True)
    server.server_close()
    print("✅ Остановлен")