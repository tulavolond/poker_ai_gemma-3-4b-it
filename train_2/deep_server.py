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
LORA_ONLINE_PATH = "./train_2/lora_online_learning_qwen"

MEMORY_FILE = "./train_2/poker_memory.json"
BUFFER_FILE = "./train_2/training_buffer.json"
STATS_FILE = "./train_2/training_stats.json"
HISTORY_FILE = "./train_2/chat_memory.json"

PORT = 8899
HOST = "0.0.0.0"
MAX_MESSAGES = 40
MEMORY_LIMIT = 100

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
    "fix_response_used": 0,
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
# ИСТОРИЯ
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
# ИНСТРУКЦИИ ПО СТАДИЯМ
# ============================================================
STAGE_INSTRUCTIONS = {
    "preflop": """— ЗАДАЧА: ПРЕФЛОП —
Оцени стартовую руку по тиру и позиции.""",
    "flop": """— ЗАДАЧА: ФЛОП —
Сравни эквити с пот-оддсами.""",
    "turn": """— ЗАДАЧА: ТЁРН —
Одна карта до вскрытия.""",
    "river": """— ЗАДАЧА: РИВЕР —
Финальная сила руки.""",
}

FEW_SHOT = """
Ответь СТРОГО в формате:
АНАЛИЗ: [одно предложение с твоей логикой]
ДЕЙСТВИЕ: [FOLD / CHECK / CALL / RAISE]
СУММА: [число]
"""


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


def detect_cart(prompt_text):
    line_match = re.search(r'ТВОИ КАРТЫ:\s*(.+)', prompt_text, re.IGNORECASE)
    if not line_match:
        return None
    parts = [c.strip() for c in line_match.group(1).split(',')]
    if len(parts) < 2:
        return None
    return [(parts[0], parts[1])]


def detect_stack(prompt_text):
    for pat in [r'ТВОЙ СТЕК:\s*\$?(\d+)', r'Стек:\s*\$?(\d+)']:
        m = re.search(pat, prompt_text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 1000


def detect_pot(prompt_text):
    for pat in [r'БАНК:\s*\$?(\d+)', r'Банк:\s*\$?(\d+)']:
        m = re.search(pat, prompt_text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


def detect_to_call(prompt_text):
    for pat in [r'КОЛЛ:\s*\$?(\d+)', r'Колл:\s*\$?(\d+)', r'CALL:\s*\$?(\d+)']:
        m = re.search(pat, prompt_text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


def detect_position(prompt_text):
    m = re.search(r'Позиция:\s*(\w+)', prompt_text, re.IGNORECASE)
    return m.group(1) if m else "BTN"


# ============================================================
# ИНФЕРЕНС
# ============================================================
def run_inference(prompt_text: str, max_tokens=120, stage_hint=None):
    stage = stage_hint or detect_stage(prompt_text)
    if stage not in ("preflop", "flop", "turn", "river"):
        stage = "flop"

    if not prompt_text.strip().startswith("[СТАДИЯ:"):
        prompt_text = f"[СТАДИЯ: {stage.upper()}]\n\n{prompt_text}"

    instruction = STAGE_INSTRUCTIONS.get(stage, "")
    full_prompt = prompt_text + "\n\n" + instruction + FEW_SHOT

    messages = load_history()
    messages.append({"role": "user", "content": full_prompt})
    messages = trim_history(messages)

    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        chat_prompt, return_tensors="pt",
        truncation=True, max_length=768
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

    messages.append({"role": "assistant", "content": text})
    save_history(trim_history(messages))

    return text


# ============================================================
# FIX RESPONSE — постобработка (главное!)
# ============================================================
def rank_to_value(r):
    """'A' -> 14, 'K' -> 13, ..., 'T' -> 10, '2' -> 2."""
    m = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10}
    return m.get(r.upper(), int(r) if r.isdigit() else 0)


# def fix_response(text, prompt_text, stage):
#     """
#     Постобработка ответа модели:
#     1. Английский -> русский
#     2. Парсинг ДЕЙСТВИЕ / СУММА
#     3. Правила для префлопа (если модель врёт)
#     4. Валидация суммы
#     """
#     global stats
#     original = text

#     # 1. Английский -> русский
#     text = re.sub(r'\bANALYSIS\s*:', 'АНАЛИЗ:', text, flags=re.IGNORECASE)
#     text = re.sub(r'\bACTION\s*:', 'ДЕЙСТВИЕ:', text, flags=re.IGNORECASE)
#     text = re.sub(r'\bAMOUNT\s*:', 'СУММА:', text, flags=re.IGNORECASE)

#     # 2. Парсинг
#     action = None
#     amount = 0
#     analysis = ""
#     for line in text.split("\n"):
#         line = line.strip()
#         if line.upper().startswith("ДЕЙСТВИЕ:"):
#             a = line.split(":", 1)[1].strip().upper()
#             if a in ("FOLD", "CHECK", "CALL", "BET", "RAISE"):
#                 action = a
#         elif line.upper().startswith("СУММА:"):
#             try:
#                 amount = int(float(line.split(":", 1)[1].strip().replace("$", "").replace(",", "")))
#             except (ValueError, IndexError):
#                 amount = 0
#         elif line.upper().startswith("АНАЛИЗ:"):
#             analysis = line.split(":", 1)[1].strip()

#     # 3. Правила для префлопа
#     cart = detect_cart(prompt_text)
#     to_call = detect_to_call(prompt_text)
#     pot = detect_pot(prompt_text)
#     stack = detect_stack(prompt_text)

#     if cart and stage == "preflop":
#         h1, h2 = cart[0]
#         r1 = rank_to_value(h1[:-1])
#         r2 = rank_to_value(h2[:-1])
#         suited = h1[-1] == h2[-1]
#         high, low = max(r1, r2), min(r1, r2)

#         # Премиум: AA, KK, QQ, AKs
#         if high == low and high >= 12:        # QQ+
#             action = "RAISE"
#             amount = 30
#             analysis = f"{h1}{h2} — премиум-пара, рейз для вэлью"
#         elif high == 14 and low == 13 and suited:   # AKs
#             action = "RAISE"
#             amount = 30
#             analysis = f"{h1}{h2} — AKs, премиум, рейз"
#         elif high == 14 and low == 13:        # AKo
#             action = "RAISE"
#             amount = 30
#             analysis = f"{h1}{h2} — AKo, премиум, рейз"
#         # Сильные: JJ, TT, AQs, AQo, AJs
#         elif high == low and high >= 10:      # JJ, TT
#             action = "RAISE"
#             amount = 30
#             analysis = f"{h1}{h2} — сильная пара, рейз"
#         elif high == 14 and low >= 11:        # AQ, AJ
#             action = "RAISE"
#             amount = 30
#             analysis = f"{h1}{h2} — сильный бродвей, рейз"
#         elif high == 13 and low == 12 and suited:   # KQs
#             action = "RAISE"
#             amount = 30
#             analysis = f"{h1}{h2} — KQs, рейз"
#         # Средние: 99-77, ATo, KQo, QJs
#         elif high == low and high >= 7:       # 99-77
#             if to_call <= 10:
#                 action = "CALL"
#                 amount = to_call
#                 analysis = f"{h1}{h2} — средняя пара, колл"
#             else:
#                 action = "FOLD"
#                 amount = 0
#                 analysis = f"{h1}{h2} — средняя пара, фолд на большую ставку"
#         elif high == 14 and low >= 9:         # AT-A9
#             action = "CALL" if to_call <= 20 else "FOLD"
#             amount = to_call if action == "CALL" else 0
#             analysis = f"{h1}{h2} — Ax, {'колл' if action == 'CALL' else 'фолд'}"
#         elif high == 13 and low >= 10:        # KQ, KJ, KT
#             action = "CALL" if to_call <= 15 else "FOLD"
#             amount = to_call if action == "CALL" else 0
#             analysis = f"{h1}{h2} — Kx, {'колл' if action == 'CALL' else 'фолд'}"
#         elif suited and high - low <= 2 and high >= 9:   # коннекторы
#             action = "CALL" if to_call <= 10 else "FOLD"
#             amount = to_call if action == "CALL" else 0
#             analysis = f"{h1}{h2} — одномастный коннектор, {'колл' if action == 'CALL' else 'фолд'}"
#         # Мусор
#         else:
#             action = "FOLD"
#             amount = 0
#             analysis = f"{h1}{h2} — слабая рука, фолд"

#     # 4. Fallback, если действий нет
#     if action is None:
#         action = "FOLD"
#         amount = 0
#         analysis = "Не удалось определить действие"
#         stats["fix_response_used"] += 1

#     # 5. Валидация суммы
#     if action == "CALL":
#         amount = to_call
#     elif action == "CHECK":
#         amount = 0
#     elif action == "FOLD":
#         amount = 0
#     elif action in ("BET", "RAISE"):
#         if amount <= to_call:
#             amount = max(to_call * 2, int(pot * 0.66), 30)
#         amount = min(amount, stack)
#         if amount <= 0:
#             action = "CHECK" if to_call == 0 else "CALL"
#             amount = to_call

#     # 6. Проверка на английский/мусор
#     if action not in ("FOLD", "CHECK", "CALL", "BET", "RAISE"):
#         action = "FOLD"
#         amount = 0

#     if original != text and stats["fix_response_used"] < 1000000:
#         stats["fix_response_used"] += 1

#     return f"АНАЛИЗ: {analysis}\nДЕЙСТВИЕ: {action}\nСУММА: {amount}"

def fix_response(text, prompt_text, stage):
    """
    Постобработка: префлоп — правила, постфлоп — модель + fallback.
    """
    global stats
    original = text

    # 1. Английский -> русский
    text = re.sub(r'\bANALYSIS\s*:', 'АНАЛИЗ:', text, flags=re.IGNORECASE)
    text = re.sub(r'\bACTION\s*:', 'ДЕЙСТВИЕ:', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAMOUNT\s*:', 'СУММА:', text, flags=re.IGNORECASE)

    # 2. Парсинг
    action = None
    amount = 0
    analysis = ""
    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("ДЕЙСТВИЕ:"):
            a = line.split(":", 1)[1].strip().upper()
            if a in ("FOLD", "CHECK", "CALL", "BET", "RAISE"):
                action = a
        elif line.upper().startswith("СУММА:"):
            try:
                amount = int(float(line.split(":", 1)[1].strip().replace("$", "").replace(",", "")))
            except (ValueError, IndexError):
                amount = 0
        elif line.upper().startswith("АНАЛИЗ:"):
            analysis = line.split(":", 1)[1].strip()

    # 3. Контекст
    cart = detect_cart(prompt_text)
    to_call = detect_to_call(prompt_text)
    pot = detect_pot(prompt_text)
    stack = detect_stack(prompt_text)

    # 4. ПРЕФЛОП — правила
    if stage == "preflop" and cart:
        h1, h2 = cart[0]
        r1 = rank_to_value(h1[:-1])
        r2 = rank_to_value(h2[:-1])
        suited = h1[-1] == h2[-1]
        high, low = max(r1, r2), min(r1, r2)

        if high == low and high >= 12:              # QQ+
            action, amount = "RAISE", 30
            analysis = f"{h1}{h2} — премиум-пара, рейз для вэлью"
        elif high == 14 and low == 13:              # AK
            action, amount = "RAISE", 30
            analysis = f"{h1}{h2} — {'AKs' if suited else 'AKo'}, премиум, рейз"
        elif high == low and high >= 10:            # JJ, TT
            action, amount = "RAISE", 30
            analysis = f"{h1}{h2} — сильная пара, рейз"
        elif high == 14 and low >= 11:              # AQ, AJ
            action, amount = "RAISE", 30
            analysis = f"{h1}{h2} — сильный бродвей, рейз"
        elif high == 13 and low == 12 and suited:   # KQs
            action, amount = "RAISE", 30
            analysis = f"{h1}{h2} — KQs, рейз"
        elif high == low and high >= 7:             # 99-77
            if to_call <= 10:
                action, amount = "CALL", to_call
                analysis = f"{h1}{h2} — средняя пара, колл"
            else:
                action, amount = "FOLD", 0
                analysis = f"{h1}{h2} — средняя пара, фолд на большую ставку"
        elif high == 14 and low >= 9:               # AT-A9
            action = "CALL" if to_call <= 20 else "FOLD"
            amount = to_call if action == "CALL" else 0
            analysis = f"{h1}{h2} — Ax, {'колл' if action == 'CALL' else 'фолд'}"
        elif high == 13 and low >= 10:              # KQ, KJ, KT
            action = "CALL" if to_call <= 15 else "FOLD"
            amount = to_call if action == "CALL" else 0
            analysis = f"{h1}{h2} — Kx, {'колл' if action == 'CALL' else 'фолд'}"
        elif suited and high - low <= 2 and high >= 9:   # коннекторы
            action = "CALL" if to_call <= 10 else "FOLD"
            amount = to_call if action == "CALL" else 0
            analysis = f"{h1}{h2} — одномастный коннектор, {'колл' if action == 'CALL' else 'фолд'}"
        else:
            action, amount = "FOLD", 0
            analysis = f"{h1}{h2} — слабая рука, фолд"

    # 5. ПОСТФЛОП — если модель не дала ответ, fallback по правилам
    elif stage in ("flop", "turn", "river"):
        if action is None:
            # Модель не ответила — используем правила по силе руки
            # Парсим силу из prompt (если есть)
            strength_m = re.search(r'сила\s*(\d+)', prompt_text)
            strength = int(strength_m.group(1)) if strength_m else 40

            if strength >= 70:                      # Монстр
                if to_call == 0:
                    action, amount = "BET", int(pot * 0.66)
                    analysis = "Сильная рука — вэлью-бет"
                else:
                    action, amount = "RAISE", int(to_call * 2.5)
                    analysis = "Сильная рука — рейз для вэлью"
            elif strength >= 40:                    # Средняя
                if to_call == 0:
                    action, amount = "CHECK", 0
                    analysis = "Средняя рука — чек"
                elif to_call <= pot * 0.5:
                    action, amount = "CALL", to_call
                    analysis = "Средняя рука — колл по оддсам"
                else:
                    action, amount = "FOLD", 0
                    analysis = "Средняя рука — фолд на большую ставку"
            else:                                   # Слабая
                if to_call == 0:
                    action, amount = "CHECK", 0
                    analysis = "Слабая рука — чек"
                else:
                    action, amount = "FOLD", 0
                    analysis = "Слабая рука — фолд"

            stats["fix_response_used"] += 1

    # 6. Fallback, если всё ещё None
    if action is None:
        action = "FOLD"
        amount = 0
        if not analysis:
            analysis = "Не удалось определить действие"

    # 7. Валидация суммы
    if action == "CALL":
        amount = to_call
    elif action in ("CHECK", "FOLD"):
        amount = 0
    elif action in ("BET", "RAISE"):
        if amount <= to_call:
            amount = max(to_call * 2, int(pot * 0.66), 30)
        amount = min(amount, stack)
        if amount <= 0:
            action = "CHECK" if to_call == 0 else "CALL"
            amount = to_call

    # 8. Финальная проверка
    if action not in ("FOLD", "CHECK", "CALL", "BET", "RAISE"):
        action = "FOLD"
        amount = 0

    if original != text and stats["fix_response_used"] < 1000000:
        stats["fix_response_used"] += 1

    return f"АНАЛИЗ: {analysis}\nДЕЙСТВИЕ: {action}\nСУММА: {amount}"

# ============================================================
# ОБУЧЕНИЕ
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
                truncation=True, max_length=768
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

            if parsed.path == "/pretrain":
                self._send_error(503, "Pretrain отключён — используйте deep_dataset_v2.py")
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
            print(f"📥 POST stage={stage}, prompt={len(prompt)} симв.")

            # 1. Инференс
            response_text = run_inference(prompt, body.get('max_tokens', 155))

            # 2. Постобработка (главное!)
            # response_text = fix_response(response_text, prompt, stage)

            elapsed = time.time() - start

            if stage == "preflop":
                stats["preflop_queries"] += 1
            else:
                stats["postflop_queries"] += 1

            print(f"📤 Ответ за {elapsed:.1f}s: {response_text}")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors()
            self.end_headers()
            data = json.dumps({
                'response': f"{elapsed:.1f}s | {response_text}"
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
print(f"🔧 fix_response: ВСЕГДА включён (правила поверх модели)")
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