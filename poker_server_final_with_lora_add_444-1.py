"""
Сервер покерного ИИ с ЛоРА и исправлениями:
- ThreadingHTTPServer (параллельная обработка)
- Правильное формирование prompt/completion для обучения
- Async-сохранение LoRA
- Few-shot промпт против галлюцинаций
- Кэш памяти
- Обработка разорванных соединений
"""
import os
import json
import traceback
import threading
import torch
import atexit
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler
from peft import LoraConfig, get_peft_model, PeftModel

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

# ============================================================
# LORA
# ============================================================
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

# ============================================================
# ОБУЧЕНИЕ
# ============================================================
# Добавьте в начало:
from trainer import generate_training_examples

# Новая команда POST для "pre-train"
def do_GET_pretrain(self):
    """Генерирует синтетические примеры для LoRA"""
    examples = generate_training_examples(5000)
    
    # Добавляем в training_buffer
    global training_buffer
    training_buffer.extend(examples)
    
    # Запускаем обучение
    result = train_batch()
    
    self.send_response(200)
    self._set_cors()
    self.end_headers()
    self.wfile.write(json.dumps({
        "status": "pretrain_done",
        "examples_generated": len(examples),
        "training_result": result
    }).encode('utf-8'))

# ============================================================
# ТОКЕНИЗАТОР И ШАБЛОН
# ============================================================
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

# ============================================================
# ПАМЯТЬ (КЭШ В ПАМЯТИ + ФАЙЛ)
# ============================================================
training_buffer = []
memory_cache = []
MEMORY_LIMIT = 30
last_loss = 0.0
training_lock = threading.Lock()  # защита от гонок при обучении
save_lock = threading.Lock()
save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LoraSaver")

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

def save_memory_to_disk():
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory_cache[-100:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Ошибка записи памяти: {e}")

def add_to_memory(hand_data):
    global memory_cache
    memory_cache = load_memory()
    memory_cache.append(hand_data)
    if len(memory_cache) > 100:
        memory_cache = memory_cache[-100:]
    save_memory_to_disk()

def save_lora_weights():
    """Синхронное сохранение (использовать осторожно)"""
    print("\n💾 Сохранение LoRA...")
    try:
        with save_lock:
            model.save_pretrained(LORA_PATH)
            tokenizer.save_pretrained(LORA_PATH)
        print(f"✅ Сохранено в {LORA_PATH}")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")

def save_lora_async():
    """Асинхронное сохранение — не блокирует сервер"""
    def _save():
        try:
            with save_lock:
                model.save_pretrained(LORA_PATH)
                tokenizer.save_pretrained(LORA_PATH)
            print("✅ LoRA async saved")
        except Exception as e:
            print(f"❌ Async save error: {e}")
    save_executor.submit(_save)

# ============================================================
# ФОРМИРОВАНИЕ ДАННЫХ ДЛЯ ОБУЧЕНИЯ
# ============================================================
def build_training_pair(data):
    """
    ИСПРАВЛЕНИЕ: формируем prompt и completion из данных руки.
    Клиент отправляет: stage, my_cards, board, pot, to_call, stack,
                      action_taken, outcome, lesson
    """
    prompt = f"""Ты профессиональный игрок в покер.
СТАДИЯ: {data.get('stage', '?')}
ТВОИ КАРТЫ: {data.get('my_cards', '?')}
ОБЩИЕ КАРТЫ: {data.get('board', '?')}
БАНК: ${data.get('pot', 0)}, ТВОЙ СТЕК: ${data.get('stack', 0)}
НУЖНО КОЛЛИРОВАТЬ: ${data.get('to_call', 0)}

Выбери ОДНО действие: FOLD, CHECK, CALL или RAISE.
Ответь СТРОГО в формате (без пояснений):
АНАЛИЗ: [1 короткое предложение]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/RAISE]
СУММА: [число, для FOLD/CHECK = 0]"""

    # completion — правильное действие + урок
    action = data.get('action_taken', 'FOLD')
    outcome = data.get('outcome', 'LOSS')
    lesson = data.get('lesson', '')
    
    # Разбираем действие
    parts = str(action).split()
    act_word = parts[0].upper() if parts else 'FOLD'
    amount = parts[1].replace('$', '') if len(parts) > 1 else '0'
    
    completion = f"""АНАЛИЗ: {lesson}
ДЕЙСТВИЕ: {act_word}
СУММА: {amount}"""

    # reward: победа = +1, поражение с правильным фолдом = +0.5, поражение = -0.3
    reward = 1.0
    if 'LOSS' in outcome:
        if 'FOLD' in action.upper():
            reward = 0.3  # фолд при проигрыше — почти правильное решение
        else:
            reward = 0.1  # проиграли с коллом/рейзом — плохо

    return {"prompt": prompt, "completion": completion, "reward": reward}

# ============================================================
# ОБУЧЕНИЕ НА БАТЧЕ (gradient accumulation)
# ============================================================
def train_batch():
    global last_loss, training_buffer
    with training_lock:
        if not optimizer or len(training_buffer) < MEMORY_LIMIT:
            return {"status": "waiting", "buffer_size": len(training_buffer)}
        
        batch = training_buffer[:]
        training_buffer = []
    
    print(f"\n🧠 ОБУЧЕНИЕ: {len(batch)} рук...")
    model.train()
    total_loss = 0.0
    
    optimizer.zero_grad()  # ОБНУЛЯЕМ ПЕРЕД БАТЧЕМ
    
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
            loss = outputs.loss * item.get('reward', 1.0) / len(batch)  # усредняем
            loss.backward()
            total_loss += loss.item() * len(batch)
        except Exception as e:
            print(f"⚠️ Пропуск примера {i}: {e}")
            continue
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    last_loss = total_loss / max(len(batch), 1)
    model.eval()
    
    # АСИНХРОННОЕ сохранение — не блокирует ответы!
    save_lora_async()
    print(f"✅ Обучено. Loss: {last_loss:.4f}")
    
    return {"status": "trained", "loss": last_loss, "buffer_size": 0}

# ============================================================
# ИНФЕРЕНС
# ============================================================
FEW_SHOT_EXAMPLE = """
ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:
АНАЛИЗ: У меня пара тузов, банк небольшой, играю агрессивно.
ДЕЙСТВИЕ: RAISE
СУММА: 50
"""
STAGE_INSTRUCTIONS = {
    "preflop": """ВАЖНО: На префлопе оценивай только категорию руки (1-8 тир).
- Тир 1-2: всегда рейз
- Тир 3-4: колл в позиции, фолд из ранней позиции  
- Тир 5-6: колл только дёшево
- Тир 7-8: фолд""",
    
    "flop": """ВАЖНО: На флопе считай:
1. Попадание (пара/сет/две пары)
2. Дро (флеш-дро = 9 аутов, OESD = 8 аутов)
3. Эквити через pot odds
Если эквити > pot odds → колл/рейз, иначе фолд.""",
    
    "turn": """ВАЖНО: На тёрне важна математика.
Pot odds = ставка / (банк + ставка)
Если твоё эквити > pot odds → колл.
Дро имеет ~20% с одной картой до ривера.""",
    
    "river": """ВАЖНО: На ривере нет дро, только ценность.
Если эквити > 60% → колл/рейз для вэлью.
Если < 25% → только блеф или фолд."""
}

def run_inference_2(prompt_text, max_tokens=100):
    # Определяем стадию из промпта
    stage = "preflop"
    for s in ["preflop", "flop", "turn", "river"]:
        if s.upper() in prompt_text.upper() or f"СТУДИЯ: {s.upper()}" in prompt_text:
            stage = s
            break
    
    instruction = STAGE_INSTRUCTIONS.get(stage, "")
    enhanced_prompt = f"{prompt_text}\n\n{instruction}"
    return enhanced_prompt

def run_inference(prompt_text, max_tokens=100):
   
    # Добавляем few-shot для устранения галлюцинаций
    full_prompt = prompt_text + FEW_SHOT_EXAMPLE
    
    messages = [{"role": "user", "content": full_prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True, 
            temperature=0.7, top_p=0.9,
            repetition_penalty=1.15,  # ↑ против повторов
            eos_token_id=stop_ids,
        )
    
    gen_ids = outputs[0][input_length:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    # Чистим от мусора
    for tok in ('<end_of_turn>', '<eos>', '<bos>', '<pad>', '<start_of_turn>', 'model\n', 'user\n'):
        text = text.replace(tok, '')
    
    # ВАЛИДАЦИЯ: если нет ДЕЙСТВИЕ — добавляем FOLD
    if 'ДЕЙСТВИЕ:' not in text:
        text = f"АНАЛИЗ: Не уверен.\nДЕЙСТВИЕ: FOLD\nСУММА: 0"
    elif 'СУММА:' not in text:
        text += "\nСУММА: 0"
    
    return text.strip()

# def reinforcement_training_cycle(games=100):
#     """
#     RL-цикл:
#     1. Модель играет против себя
#     2. Победители получают reward +1
#     3. Проигравшие -0.3
#     4. Обучаемся на исходах
#     """
#     for game in range(games):
#         # ИИ играет сам с собой через ваш HTML клиент
#         # или через автоматический симулятор
        
#         # После каждой руки:
#         for player_decision in game_history:
#             reward = 1.0 if player_decision.won else -0.3
            
#             training_buffer.append({
#                 "prompt": player_decision.prompt,
#                 "completion": player_decision.actual_action,
#                 "reward": reward
#             })
        
#         # Каждые 30 рук — обучаем
#         if len(training_buffer) >= MEMORY_LIMIT:
#             train_batch()
            
# ============================================================
# HTTP-ОБРАБОТЧИК
# ============================================================
class ModelHandler(BaseHTTPRequestHandler):
    # Увеличиваем timeout
    timeout = 60

    def _set_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _safe_write(self, data):
        """Безопасная запись с проверкой соединения"""
        try:
            self.wfile.write(data)
            return True
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return False
        except Exception as e:
            print(f"⚠️ Write error: {e}")
            return False

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        try:
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
                'recent_memory': memory
            }, ensure_ascii=False).encode('utf-8')
            self._safe_write(data)
        except Exception as e:
            self._send_error(500, str(e))

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._send_error(400, "Empty body")
                return
            
            raw_body = self.rfile.read(length)
            try:
                body = json.loads(raw_body.decode('utf-8'))
            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON")
                return

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
                print("⚠️ Клиент разорвал соединение до получения ответа")

        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            print("⚠️ Клиент разорвал соединение (это нормально при timeout)")
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
            
            # Сохраняем в память
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
            
            # ИСПРАВЛЕНИЕ: формируем prompt/completion правильно!
            training_pair = build_training_pair(hand_data)
            training_buffer.append(training_pair)
            
            print(f"📝 В буфере: {len(training_buffer)}/{MEMORY_LIMIT}")
            
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
            pass  # сокет уже мёртв — игнорируем

    def log_message(self, format, *args):
        if 'OPTIONS' not in str(args[0]):
            print(f"🌐 {args[0]}")


# ============================================================
# МНОГОПОТОЧНЫЙ СЕРВЕР
# ============================================================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Обрабатывает каждый запрос в отдельном потоке"""
    daemon_threads = True
    allow_reuse_address = True

server = ThreadedHTTPServer((HOST, PORT), ModelHandler)

print(f"\n🚀 Сервер запущен (МНОГОПОТОЧНЫЙ): http://{HOST}:{PORT}")
print(f"🧠 Память: {MEMORY_FILE}")
print(f"🎯 LoRA: {LORA_PATH}")
print("Нажмите Ctrl+C для остановки.\n")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Остановка...")
    save_lora_weights()
    save_executor.shutdown(wait=False)
    server.server_close()
    print("✅ Сервер остановлен.")