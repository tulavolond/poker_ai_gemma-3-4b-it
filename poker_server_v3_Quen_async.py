"""
poker_server_v3.py - Покерный сервер с обученной моделью.
ОДНА модель (Qwen2.5-1.5B) для всех стадий.
Работает на 8GB видеопамяти.

Запуск: python poker_server_v3.py
"""
import os
import re
import string
import json
import traceback
import threading
import torch
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler
from peft import PeftModel, LoraConfig, get_peft_model
from analyzer import classify_preflop_hand, calculate_equity, build_deck, analyze_preflop, classify_preflop_hand, RANKS, SUITS, RANK_NAMES, Card, evaluate_hand, HAND_STRENGTH_BASE

# Импортируем наш тренер для генерации данных
from trainer import generate_training_examples

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ============================================================
# КОНФИГУРАЦИЯ - ВСЕ НАСТРОЙКИ ЗДЕСЬ
# ============================================================
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PREFLOP_PATH = "./lora_preflop_stage_3" # <- Сгенерирована train_stage_3_google.py и выложена в ollama взять можно chat_memory.py
# LORA_PREFLOP_PATH = "./lora_online_learning" # <- Сгенерирована онлайн Invoke-RestMethod -Uri http://192.168.1.103:8899/pretrain
LORA_ONLINE_PATH = "./lora_online_learning"
MEMORY_FILE = "./poker_memory.json"
BUFFER_FILE = "./training_buffer.json"
STATS_FILE = "./training_stats.json"
PORT = 8899
HOST = "192.168.1.103"

MEMORY_LIMIT = 100  # сколько примеров копить до обучения

print("=" * 60)
print("  ЗАГРУЗКА МОДЕЛИ")
print("=" * 60)

# ============================================================
# 1. ЗАГРУЗКА БАЗОВОЙ МОДЕЛИ
# ============================================================
print(f"📦 Загрузка модели: {BASE_MODEL_ID}")
print("   Это займёт 1-2 минуты при первом запуске...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
    padding_side="right",
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
print(f"   Память GPU занята: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================
# 2. ЗАГРУЗКА ОБУЧЕННОЙ LoRA ДЛЯ ПРЕФЛОПА
# ============================================================
PREFLOP_AVAILABLE = False
preflop_lora_loaded = False

if os.path.exists(LORA_PREFLOP_PATH):
    try:
        print(f"\n🎯 Загрузка обученной префлоп-модели из {LORA_PREFLOP_PATH}...")
        model = PeftModel.from_pretrained(model, LORA_PREFLOP_PATH)
        model.eval()
        PREFLOP_AVAILABLE = True
        preflop_lora_loaded = True
        print("✅ Обученная префлоп-модель загружена")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки префлоп-модели: {e}")
        PREFLOP_AVAILABLE = False
else:
    print(f"\n⚠️ Префлоп-модель не найдена: {LORA_PREFLOP_PATH}")
    print("   Используется базовая модель для всех стадий")

# ============================================================
# 3. ПАМЯТЬ, БУФЕР, СТАТИСТИКА
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
    """Загружает все сохранённые данные с диска."""
    global training_buffer, memory_cache, stats

    # Память рук
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                memory_cache = json.load(f)
            print(f"📂 Память рук: {len(memory_cache)} записей")
        except:
            memory_cache = []

    # Буфер обучения
    if os.path.exists(BUFFER_FILE):
        try:
            with open(BUFFER_FILE, 'r', encoding='utf-8') as f:
                training_buffer = json.load(f)
            print(f"📂 Буфер обучения: {len(training_buffer)} примеров")
        except:
            training_buffer = []

    # Статистика
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            print(f"📂 Статистика: {stats}")
        except:
            pass


def save_all_data():
    """Сохраняет все данные на диск."""
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory_cache[-100:], f, ensure_ascii=False, indent=1)
        with open(BUFFER_FILE, 'w', encoding='utf-8') as f:
            json.dump(training_buffer, f, ensure_ascii=False, indent=1)
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Ошибка сохранения данных: {e}")


def add_to_memory(hand_data):
    global memory_cache
    memory_cache.append(hand_data)
    if len(memory_cache) > 100:
        memory_cache = memory_cache[-100:]
    save_all_data()


# ============================================================
# 4. ПРОМПТЫ ПОД КАЖДУЮ СТАДИЮ
# ============================================================
STAGE_INSTRUCTIONS = {
    "preflop": """— ЗАДАЧА: ПРЕФЛОП —
Оцени стартовую руку по тиру и позиции.
Правила:
- Тир 1-2: всегда рейз или 3-бет
- Тир 3-4: рейз из средней/поздней позиции
- Тир 5-6: только дешёвый вход из поздней позиции
- Тир 7: фолд""",

    "flop": """— ЗАДАЧА: ФЛОП (3 карты на столе) —
Сравни эквити с пот-оддсами.
Правила:
- Сильное попадание (сет/2 пары): вэлью-бет
- Топ-пара с эквити > 50%: продолженная ставка или колл
- Флеш-дро или стрит-дро: можно продолжать
- Эквити < 30%: фолд""",

    "turn": """— ЗАДАЧА: ТЁРН (4 карты на столе) —
Одна карта до вскрытия.
Правила:
- Сильная комбинация: вэлью-бет
- Дро: ~20% шанс, считай пот-оддсы
- Если эквити < пот-оддсов: фолд""",

    "river": """— ЗАДАЧА: РИВЕР (5 карт на столе) —
Дро больше нет, только финальная сила.
Правила:
- Эквити > 65%: колл или вэлью-рейз
- Эквити 30-65%: колл если пот-оддсы позволяют
- Эквити < 30%: фолд или редкий блеф"""
}

FEW_SHOT = """

ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:
АНАЛИЗ: У меня пара тузов, эквити 78%, вэлью-бет.
ДЕЙСТВИЕ: RAISE
СУММА: 50
"""

def detect_stage(prompt_text):
    """Определяет стадию из промпта."""
    p = prompt_text.upper()
    if "ПРЕФЛОП" in p or "PRE-FLOP" in p or "PRE-FLOP" in p:
        return "preflop"
    if "ФЛОП" in p or "FLOP" in p:
        return "flop"
    if "ТЁРН" in p or "TURN" in p:
        return "turn"
    if "РИВЕР" in p or "RIVER" in p:
        return "river"
    return "unknown"

def detect_cart(prompt_text):
    # 1. Ищем строку с картами
    line_match = re.search(r'ТВОИ КАРТЫ: (.+)', prompt_text.upper())
    
    # 2. Если нашли — извлекаем, если нет — возвращаем пустой список
    if line_match:
        # Сразу распаковываем в две переменные
        card1, card2 = [card.strip() for card in line_match.group(1).split(',')]
        
        # Формируем список с одним кортежем
        premium_hands = [(card1, card2)]
        
        # (Опционально) Для отладки можно оставить print, но лучше залогировать
        # print(f"cards: {premium_hands}")
        
        return premium_hands
    else:
        # Возвращаем пустой список, а не список с пустым кортежем
        return None
    
def detect_stack(prompt_text):
    # 1. Ищем строку с картами
    line_match = re.search(r'ТВОЙ СТЕК: (.+)', prompt_text.upper())
    if line_match:
        line_match = line_match.group(1).replace('$', '')
        return int(line_match)
    else:
        # Возвращаем пустой список, а не список с пустым кортежем
        return 1010

def generate_response(prompt, max_new_tokens=100):
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
            do_sample=False,  # Детерминированный ответ
            temperature=1.0,
            repetition_penalty=1.1,
        )
    
    # Извлекаем только сгенерированную часть
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Чистим от мусора
    response = response.strip()
    for tok in ('<|im_end|>', '<|im_start|>', 'assistant', 'user'):
        response = response.replace(tok, '')
    # Валидация формата
    if 'ДЕЙСТВИЕ:' not in response:
        response = f"АНАЛИЗ: ИИ не дал ответа!\nДЕЙСТВИЕ: FOLD\nСУММА: 0"
    elif 'СУММА:' not in response:
        response += "\nСУММА: 0"
    
    return response.strip()

# ============================================================
# 5. ИНФЕРЕНС
# ============================================================
def run_inference(prompt_text, max_tokens=100, stage_hint=None):
    stage = stage_hint or detect_stage(prompt_text)
    stack = detect_stack(prompt_text)
    print(f"stage: {stage} stack: {stack}")
    cart = detect_cart(prompt_text)
    h1_str, h2_str = cart[0]
    rank_map = {str(i): i for i in range(2, 11)}  # '2': 2, ..., '10': 10
    rank_map.update({v: k for k, v in RANK_NAMES.items()})
    # Безопасное разделение: ранг — всё до масти, масть — последний символ
    h1 = Card(rank_map[h1_str[:-1]], h1_str[-1])
    h2 = Card(rank_map[h2_str[:-1]], h2_str[-1])
    print(f"h1: {h1}, h2: {h2}")
    # 1. ПРОВЕРКА: Если список пустой, выходим или возвращаем ошибку
    # if (stage == "preflop" or stage == 'flop' or stage == 'turn') and cart != None:
    if (stage == "preflop") and cart != None:
        print(f"\n🃏 Рука: {h1}, {h2}")
        analysis = analyze_preflop(
            hero_cards=[h1, h2],
            position="BTN (баттон)",
            action_before="Все сбросили до тебя",
            stack=stack,
            bb=10
        )
        response = generate_response(analysis["prompt"])
        print(f"🤖 Ответ модели:\n{response}")
        return h1_str + h2_str + "|" + response

    # Добавляем инструкции по стадии
    instruction = STAGE_INSTRUCTIONS.get(stage, "")
    full_prompt = prompt_text + "\n\n" + instruction + FEW_SHOT

    # Токенизация
    messages = [{"role": "user", "content": full_prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        chat_prompt, return_tensors="pt",
        truncation=True, max_length=2048
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    # Генерация
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1 if stage == "preflop" else 0.3,
            top_p=0.5,
            repetition_penalty=1.15,
        )

    # Декодирование
    gen_ids = outputs[0][input_length:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Чистка
    for tok in ('<|im_end|>', '<|im_start|>', 'assistant', 'user', '<end_of_turn>'):
        text = text.replace(tok, '')

    # Валидация формата
    if 'ДЕЙСТВИЕ:' not in text:
        text = f"АНАЛИЗ: Не уверен, играю осторожно.\nДЕЙСТВИЕ: FOLD\nСУММА: 0"
    elif 'СУММА:' not in text:
        text += "\nСУММА: 0"
        
    cart = detect_cart(prompt_text)
    # print(prompt_text)
    # 1. ПРОВЕРКА: Если список пустой, выходим или возвращаем ошибку
    if cart == None:
        print("❌ Ошибка: Карты не найдены в промпте. Инференс прерван.")
        return text.strip()
        # return "Ошибка: не удалось определить карты."
    # 2. Если список не пустой, берем ПЕРВУЮ руку (индекс 0)
    # Это решает проблему, если detect_cart вернет много рук
    try: 
        return h1_str + h2_str + "|" + text.strip()
    except:
        return text.strip()


# ============================================================
# 6. ОБУЧЕНИЕ (СОХРАНЕНО ИЗ СТАРОЙ ВЕРСИИ)
# ============================================================
def train_batch():
    """Обучение на накопленных примерах."""
    global last_loss, training_buffer, stats

    with training_lock:
        if len(training_buffer) < MEMORY_LIMIT:
            return {"status": "waiting", "buffer_size": len(training_buffer)}

        batch = training_buffer[:]
        training_buffer = []
        save_all_data()

    print(f"\n🧠 ОБУЧЕНИЕ: {len(batch)} примеров...")
    model.train()
    total_loss = 0.0

    # Включаем градиенты для LoRA параметров
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5, weight_decay=0.01
    )

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
            enc = tokenizer(
                train_text, return_tensors="pt",
                truncation=True, max_length=1500
            ).to(model.device)

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

    # Асинхронное сохранение
    save_executor.submit(save_lora_weights)

    print(f"✅ Обучено. Loss: {last_loss:.4f}")
    save_all_data()

    return {"status": "trained", "loss": last_loss, "buffer_size": 0}


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


def save_lora_weights():
    """Сохранение весов."""
    try:
        with save_lock:
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(LORA_ONLINE_PATH)
                tokenizer.save_pretrained(LORA_ONLINE_PATH)
            print(f"✅ LoRA сохранены в {LORA_ONLINE_PATH}")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения: {e}")


# ============================================================
# 7. HTTP-ОБРАБОТЧИК
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
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self._set_cors()
                self.end_headers()
                data = json.dumps({
                    'status': 'ok',
                    'preflop_model_available': PREFLOP_AVAILABLE,
                    'buffer_size': len(training_buffer),
                    'last_loss': last_loss,
                    'recent_memory': memory_cache[-100:],
                    'stats': stats
                }, ensure_ascii=False).encode('utf-8')
                self._safe_write(data)
                return

            if self.path == "/pretrain":
                self.handle_pretrain()
                return

            # Обычный GET
            # self.send_response(200)
            # self._set_cors()
            # self.end_headers()
            # self._safe_write(b"Server is running")
            memory = load_memory()[-100:]
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
        """Генерация обучающих данных."""
        try:
            print("\n🎓 [PRETRAIN] Запуск предобучения...")
            examples = generate_training_examples(1000)
            stats["pretrain_examples"] += len(examples)
            training_buffer.extend(examples)
            save_all_data()
            print(f"✅ Сгенерировано {len(examples)} примеров")

            result = train_batch()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            self._safe_write(json.dumps({
                "status": "pretrain_done",
                "examples": len(examples),
                "result": result
            }, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            print(f"❌ Pretrain error: {e}")
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

            response_text = run_inference(prompt, body.get('max_tokens', 100))

            print(f"\n Промпт: \n{prompt}\n")
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
            stats["online_examples"] += 1

            # Формируем обучающий пример
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
                "prompt": prompt,
                "completion": completion,
                "reward": reward,
                "stage": stage
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
# 8. ЗАПУСК СЕРВЕРА
# ============================================================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# Загружаем сохранённые данные
print("\n📂 Загрузка сохранённых данных...")
load_all_data()

server = ThreadedHTTPServer((HOST, PORT), ModelHandler)

print()
print("=" * 60)
print("🚀 СЕРВЕР ЗАПУЩЕН")
print("=" * 60)
print(f"📡 Адрес: http://{HOST}:{PORT}")
print(f"🎯 Префлоп-модель: {'✅ Обученная' if PREFLOP_AVAILABLE else '⚠️ Базовая'}")
print(f"📊 Статистика: http://{HOST}:{PORT}/stats")
print(f"🎓 Pretrain: http://{HOST}:{PORT}/pretrain")
print()
print("Нажмите Ctrl+C для остановки.")
print()

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Остановка сервера...")
    print("💾 Сохранение данных...")
    save_all_data()
    save_lora_weights()
    save_executor.shutdown(wait=True)
    server.server_close()
    print("✅ Сервер остановлен. Все данные сохранены.")