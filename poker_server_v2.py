"""
poker_server_v2.py - Сервер с обучением LoRA на 4 стадиях покера.
Запуск: python poker_server_v2.py
"""
import os
import json
import traceback
import threading
import torch
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler
from peft import LoraConfig, get_peft_model, PeftModel

# Импортируем наш тренер
from trainer import generate_training_examples

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "google/gemma-3-4b-it"
LORA_PATH = "./poker_gemma3_lora"
MEMORY_FILE = "./poker_memory.json"
PORT = 8899
HOST = "192.168.1.103"

print("=" * 60)
print("  ЗАГРУЗКА МОДЕЛИ")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
device_map = {"": 0} if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map=device_map,
    dtype=torch.bfloat16, token=HF_TOKEN, low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

# ============ LoRA ============
if os.path.exists(LORA_PATH):
    print(f"📂 Загружаю LoRA из {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    print("✅ LoRA загружен.")
else:
    print(f"🆕 Создаю новый LoRA...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    os.makedirs(LORA_PATH, exist_ok=True)

for name, param in model.named_parameters():
    param.requires_grad = "lora" in name.lower()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"✅ Обучаемых параметров: {trainable_params:,} / {all_params:,}")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5, weight_decay=0.01
)

# ============ Шаблон ============
GEMMA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}{{ '<start_of_turn>model\n' }}"
    "{% else %}{{ '<start_of_turn>' + message['role'] + '\n' }}{% endif %}"
    "{{ message['content'] + '<end_of_turn>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
)
if tokenizer.chat_template is None:
    tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
stop_ids = sorted(set([x for x in [tokenizer.eos_token_id, end_of_turn_id] if x is not None]))

print("✅ Модель готова.")

# ============ Память ============
training_buffer = []
memory_cache = []
MEMORY_LIMIT = 30
last_loss = 0.0
training_lock = threading.Lock()
save_lock = threading.Lock()
save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LoraSaver")
stats = {"pretrain_examples": 0, "online_examples": 0, "batches_trained": 0}


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
        print(f"❌ Ошибка записи: {e}")


def save_lora_async():
    def _save():
        try:
            with save_lock:
                model.save_pretrained(LORA_PATH)
                tokenizer.save_pretrained(LORA_PATH)
            print("✅ LoRA async saved")
        except Exception as e:
            print(f"❌ Save error: {e}")
    save_executor.submit(_save)


def save_lora_sync():
    try:
        with save_lock:
            model.save_pretrained(LORA_PATH)
            tokenizer.save_pretrained(LORA_PATH)
        print(f"✅ Сохранено в {LORA_PATH}")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")


# ============ ОБУЧЕНИЕ ============
def train_batch():
    global last_loss, training_buffer, stats
    with training_lock:
        if not optimizer or len(training_buffer) < MEMORY_LIMIT:
            return {"status": "waiting", "buffer_size": len(training_buffer)}
        batch = training_buffer[:]
        training_buffer = []

    print(f"\n🧠 ОБУЧЕНИЕ: {len(batch)} рук...")
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, item in enumerate(batch):
        try:
            messages = [
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": item['completion']}
            ]
            train_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            enc = tokenizer(train_text, return_tensors="pt", truncation=True, max_length=1500).to(model.device)

            user_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": item['prompt']}],
                tokenize=False, add_generation_prompt=True
            )
            user_len = len(tokenizer(user_text, add_special_tokens=False)["input_ids"])

            labels = enc["input_ids"].clone()
            labels[:, :user_len] = -100

            outputs = model(input_ids=enc["input_ids"], labels=labels)
            loss = outputs.loss * item.get('reward', 1.0) / len(batch)
            loss.backward()
            total_loss += loss.item() * len(batch)
        except Exception as e:
            print(f"⚠️ Пропуск примера {i}: {e}")
            continue

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    last_loss = total_loss / max(len(batch), 1)
    model.eval()
    stats["batches_trained"] += 1

    save_lora_async()
    print(f"✅ Обучено. Loss: {last_loss:.4f}")

    return {"status": "trained", "loss": last_loss, "buffer_size": 0}


# ============ ИНФЕРЕНС ============
STAGE_INSTRUCTIONS = {
    "preflop": """ВАЖНО: На префлопе оценивай только категорию руки (тир 1-8).
- Тир 1-2: премиум (AA, KK, AKs) → всегда рейз или 3-бет
- Тир 3-4: играбельные (JJ, suited connectors) → колл или рейз из позиции
- Тир 5-6: маргинальные (одномастные, Ax) → колл только дёшево
- Тир 7-8: мусор → фолд""",

    "flop": """ВАЖНО: На флопе оценивай попадания и дро:
- Сет/две пары: рейз для вэлью
- Пара с эквити > 50%: колл или чек
- Флеш-дро (9 аутов, ~35%) или OESD (8 аутов, ~32%): колл с implied odds
- Эквити < 30%: фолд""",

    "turn": """ВАЖНО: На тёрне считай pot odds:
Pot odds = ставка / (банк + ставка)
Если эквити > pot odds + 5% → колл.
Дро с 1 картой: ~20% на ривере.
Сильная рука → рейз 2-3x ставки.""",

    "river": """ВАЖНО: На ривере только ценность или блеф.
Эквити > 65% → колл или рейз для вэлью.
Эквити < 25% → фолд или редкий блеф.
Никогда не коллируй со старшей картой."""
}

FEW_SHOT = """

ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:
АНАЛИЗ: У меня пара тузов, эквити 78%, вэлью-бет.
ДЕЙСТВИЕ: RAISE
СУММА: 50
"""


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
    return "flop"


def run_inference(prompt_text, max_tokens=100):
    stage = detect_stage(prompt_text)
    instruction = STAGE_INSTRUCTIONS.get(stage, "")
    full_prompt = prompt_text + "\n\n" + instruction + FEW_SHOT

    messages = [{"role": "user", "content": full_prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=0.7, top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=stop_ids,
        )

    gen_ids = outputs[0][input_length:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    for tok in ('<end_of_turn>', '<eos>', '<bos>', '<pad>', '<start_of_turn>', 'model\n', 'user\n'):
        text = text.replace(tok, '')

    if 'ДЕЙСТВИЕ:' not in text:
        text = f"АНАЛИЗ: Не уверен, играю осторожно.\nДЕЙСТВИЕ: FOLD\nСУММА: 0"
    elif 'СУММА:' not in text:
        text += "\nСУММА: 0"

    return text.strip()


# ============ HTTP ============
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
            if self.path == "/pretrain":
                return self.handle_pretrain()
            if self.path == "/stats":
                return self.handle_stats()

            memory = load_memory()[-50:]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            data = json.dumps({
                'status': 'ok',
                'lora_active': True,
                'buffer_size': len(training_buffer),
                'last_loss': last_loss,
                'recent_memory': memory,
                'stats': stats
            }, ensure_ascii=False).encode('utf-8')
            self._safe_write(data)
        except Exception as e:
            self._send_error(500, str(e))

    def handle_pretrain(self):
        """Генерирует обучающие примеры и обучает LoRA."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self._set_cors()
        self.end_headers()

        try:
            print("\n🎓 ЗАПУСК ПРЕД-ОБУЧЕНИЯ...")
            examples = generate_training_examples(1000)
            stats["pretrain_examples"] += len(examples)
            training_buffer.extend(examples)
            print(f"✅ Сгенерировано {len(examples)} примеров")

            result = train_batch()

            self._safe_write(json.dumps({
                "status": "pretrain_done",
                "examples": len(examples),
                "result": result,
                "stats": stats
            }, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            print(f"❌ Pretrain error: {e}")
            self._safe_write(json.dumps({"error": str(e)}).encode('utf-8'))

    def handle_stats(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self._set_cors()
        self.end_headers()
        self._safe_write(json.dumps({
            "stats": stats,
            "buffer_size": len(training_buffer),
            "last_loss": last_loss
        }, ensure_ascii=False).encode('utf-8'))

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

            response_text = run_inference(prompt, body.get('max_tokens', 100))

            print(f"\n📥 Запрос. Промпт: {len(prompt)} симв.")
            print(f"📤 Ответ: {response_text[:300]}")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            data = json.dumps({'response': response_text}, ensure_ascii=False).encode('utf-8')
            if not self._safe_write(data):
                print("⚠️ Клиент разорвал соединение")

        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            print("⚠️ Клиент разорвал соединение")
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

            # Формируем обучающую пару из данных реальной игры
            stage = body.get("stage", "flop").lower()
            action = body.get("action_taken", "FOLD")
            outcome = body.get("outcome", "LOSS")
            lesson = body.get("lesson", "")

            prompt = f"""СТУДИЯ: {stage.upper()}
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
            stats["online_examples"] += 1

            training_buffer.append({
                "prompt": prompt,
                "completion": completion,
                "reward": reward,
                "stage": stage
            })

            print(f"📝 В буфере: {len(training_buffer)}/{MEMORY_LIMIT} (всего: {stats['online_examples']})")

            train_status = train_batch()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            self._safe_write(json.dumps(train_status, ensure_ascii=False).encode('utf-8'))

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


# ============ Запуск ============
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


server = ThreadedHTTPServer((HOST, PORT), ModelHandler)

print(f"\n🚀 Сервер запущен: http://{HOST}:{PORT}")
print(f"🎓 Pretrain: http://{HOST}:{PORT}/pretrain")
print(f"📊 Статистика: http://{HOST}:{PORT}/stats")
print(f"🧠 Память: {MEMORY_FILE}")
print(f"🎯 LoRA: {LORA_PATH}")
print("Нажмите Ctrl+C для остановки.\n")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Остановка...")
    save_lora_sync()
    save_executor.shutdown(wait=False)
    server.server_close()
    print("✅ Сервер остановлен.")