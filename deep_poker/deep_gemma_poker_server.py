"""
deep_poker_server.py — покерный нейро-сервер для deep_poker.html
База: рабочая конфигурация Gemma-3-4b-it (NF4 4-bit, bf16, device_map={"":0}).

ЧЕГО ЗДЕСЬ СОЗНАТЕЛЬНО НЕТ (именно это ломало GTX 1070 / Pascal):
  • prepare_model_for_kbit_training()  — переводит embedding/lm_head в fp32 (+1.25 ГиБ на словаре 262k)
  • torch.autocast(bfloat16)           — на sm_61 кидает RuntimeError / эмулируется и жрёт кэш весов
  • set_per_process_memory_fraction()  — превращал нехватку в OOM (включается только по env, по умолчанию выкл)
  • expandable_segments                — на Windows/WDDM не работает

Запуск:  python deep_poker_server.py
Эндпоинты: POST /  ·  GET /stats  ·  PUT /   (+ /save /buffer /memory /selftest /pretrain)
"""
import os

# единственная настройка аллокатора, которая реально работает на Windows
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import re, gc, json, time, threading, traceback
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    BNB_OK = True
except Exception:
    BitsAndBytesConfig, BNB_OK = None, False
try:
    from peft import PeftModel, LoraConfig, get_peft_model
    PEFT_OK = True
except Exception:
    PEFT_OK = False
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _env(k, d):  return os.getenv(k, d)
def _envb(k, d): return str(os.getenv(k, str(d))).lower() in ("1", "true", "yes", "on")
def _envi(k, d): return int(os.getenv(k, str(d)))
def _envf(k, d): return float(os.getenv(k, str(d)))


# ============================== КОНФИГ ==============================
MODEL_ID  = _env("POKER_MODEL", "google/gemma-3-4b-it")
HF_TOKEN  = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
LORA_PATH = _env("POKER_LORA", "./gemma/poker_gemma_lora")
DATA_DIR  = _env("POKER_DATA_DIR", "./gemma")
HOST      = _env("POKER_HOST", "0.0.0.0")
PORT      = _envi("POKER_PORT", 8899)

MEMORY_FILE = os.path.join(DATA_DIR, "poker_memory.json")
BUFFER_FILE = os.path.join(DATA_DIR, "training_buffer.json")
STATS_FILE  = os.path.join(DATA_DIR, "training_stats.json")

IS_GEMMA = "gemma" in MODEL_ID.lower()

USE_CHAT   = _envb("POKER_CHAT", True)        # Gemma-it отвечает только внутри chat-шаблона
USE_INSTR  = _envb("POKER_INSTR", True)       # STAGE_INSTRUCTIONS + FEW_SHOT (как в твоём рабочем сервере)
DO_SAMPLE  = _envb("POKER_SAMPLE", True)
TEMPERATURE = _envf("POKER_TEMP", 0.7)
TOP_P       = _envf("POKER_TOP_P", 0.9)
REP_PEN     = _envf("POKER_REP", 1.15)

QUANT       = _env("POKER_QUANT", "4bit").lower()   # 4bit | 8bit | none
MAX_SEQ_LEN   = _envi("POKER_SEQ", 2048)            # инференс
TRAIN_SEQ_LEN = _envi("POKER_TRAIN_SEQ", 768)       # обучение (реальные промпты ~450 токенов)
MAX_NEW       = _envi("POKER_NEW", 192)
MEMORY_LIMIT  = _envi("POKER_BUFFER", 30)
MAX_BATCH     = _envi("POKER_MAX_BATCH", 30)
LR            = _envf("POKER_LR", 1e-5)
TRAIN_LOSSES  = _envb("POKER_TRAIN_LOSSES", True)   # как в твоём сервере: reward 1.0 WIN / 0.2 LOSS
CREATE_LORA   = _envb("POKER_CREATE_LORA", True)
FORCE_ADAPTER = _envb("POKER_FORCE_ADAPTER", False)
SELFTEST      = _envb("POKER_SELFTEST", True)
VRAM_GB       = _envf("POKER_VRAM_GB", 0)           # 0 = БЕЗ лимита (рекомендуется)
EMPTY_CACHE   = _envb("POKER_EMPTY_CACHE", True)    # только после обучения

os.makedirs(DATA_DIR, exist_ok=True)

# ============================== СОСТОЯНИЕ ==============================
model = tokenizer = None
LORA_AVAILABLE = QUANT_ACTIVE = False
MODEL_DTYPE = None
STOP_IDS, BAD_IDS = [], []
END_TURN = None

training_buffer, memory_cache = [], []
last_loss = 0.0
stats = {"pretrain_examples": 0, "online_examples": 0, "batches_trained": 0,
         "preflop_queries": 0, "postflop_queries": 0, "post_requests": 0,
         "put_requests": 0, "skipped_losses": 0, "train_errors": 0,
         "bad_format": 0, "rejected_503": 0, "truncated": 0, "skipped_examples": 0}

DATA_LOCK  = threading.Lock()
MODEL_LOCK = threading.RLock()          # одна модель — один вход (generate ИЛИ train)
TRAIN_EVENT, STOP_EVENT = threading.Event(), threading.Event()
trainer = {"running": False, "last_result": None, "reported": True,
           "started": time.time(), "last_train_ts": 0.0}
pretrain_status = {"running": False, "progress": 0, "message": "Не запущен"}


def vram_info():
    if not torch.cuda.is_available():
        return None
    return {"alloc": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved": round(torch.cuda.memory_reserved() / 1024**3, 2)}


def free_vram():
    if EMPTY_CACHE and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


# ============================== ФАЙЛЫ ==============================
def _finite(o):
    if isinstance(o, float):
        return 0.0 if (o != o or o in (float("inf"), float("-inf"))) else round(o, 6)
    if isinstance(o, dict):  return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):  return [_finite(v) for v in o]
    return o


def jdump(o):
    return json.dumps(_finite(o), ensure_ascii=False, allow_nan=False)


def _atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_finite(obj), f, ensure_ascii=False, indent=1, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def save_all_data():
    with DATA_LOCK:
        try:
            _atomic(MEMORY_FILE, memory_cache[-1000:])
            _atomic(BUFFER_FILE, training_buffer)
            _atomic(STATS_FILE, stats)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения: {e}")


def load_all_data():
    global training_buffer, memory_cache
    for path, tag in ((MEMORY_FILE, "m"), (BUFFER_FILE, "b"), (STATS_FILE, "s")):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if tag == "m" and isinstance(data, list):
                memory_cache = data; print(f"📂 Память: {len(data)}")
            elif tag == "b" and isinstance(data, list):
                training_buffer = [x for x in data if isinstance(x, dict) and "prompt" in x]
                print(f"📂 Буфер: {len(training_buffer)}")
            elif tag == "s" and isinstance(data, dict):
                stats.update(data); print("📂 Статистика загружена")
        except Exception as e:
            print(f"⚠️ Не читается {path}: {e}")


# ============================== ФОРМАТ ==============================
_STAGE = {"PREFLOP": "preflop", "PRE-FLOP": "preflop", "PRE FLOP": "preflop",
          "ПРЕФЛОП": "preflop", "ПРЕ-ФЛОП": "preflop", "FLOP": "flop", "ФЛОП": "flop",
          "TURN": "turn", "ТЁРН": "turn", "ТЕРН": "turn", "RIVER": "river", "РИВЕР": "river"}


def detect_stage(text):
    p = (text or "").upper()
    m = re.search(r"\[\s*СТАДИЯ\s*:\s*([A-ZА-ЯЁ\- ]+?)\s*\]", p)
    if m and m.group(1).strip() in _STAGE:
        return _STAGE[m.group(1).strip()]
    m = re.search(r"СТ[АУ][ДЛ]ИЯ\s*:\s*([A-ZА-ЯЁ\- ]+)", p)   # СТАДИЯ + опечатка СТУДИЯ
    if m:
        for k, v in _STAGE.items():
            if m.group(1).strip().startswith(k):
                return v
    for group in (("ПРЕФЛОП", "PRE-FLOP", "PRE FLOP", "PREFLOP"), ("РИВЕР", "RIVER"),
                  ("ТЁРН", "ТЕРН", "TURN"), ("ФЛОП", "FLOP")):
        if any(k in p for k in group):
            return _STAGE[group[0]]
    return "flop"


STAGE_INSTRUCTIONS = {
 "preflop": "ВАЖНО: На префлопе оценивай только категорию руки (тир 1-8).\n"
            "- Тир 1-2: премиум (AA, KK, AKs) → рейз или 3-бет\n"
            "- Тир 3-4: играбельные (JJ, suited connectors) → колл/рейз из позиции\n"
            "- Тир 5-6: маргинальные → колл только дёшево\n- Тир 7-8: мусор → фолд",
 "flop": "ВАЖНО: На флопе оценивай попадания и дро:\n- Сет/две пары: рейз для вэлью\n"
         "- Пара с эквити > 50%: колл или чек\n- Флеш-дро (~35%) или OESD (~32%): колл с implied odds\n"
         "- Эквити < 30%: фолд",
 "turn": "ВАЖНО: На тёрне считай pot odds = ставка / (банк + ставка).\n"
         "Если эквити > pot odds + 5% → колл. Дро с 1 картой ~20%. Сильная рука → рейз 2-3x.",
 "river": "ВАЖНО: На ривере только вэлью или блеф.\nЭквити > 65% → колл/рейз. "
          "Эквити < 25% → фолд. Никогда не коллируй со старшей картой.",
}
FEW_SHOT = ("\n\nПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:\nАНАЛИЗ: У меня пара тузов, эквити 78%, вэлью-бет.\n"
            "ДЕЙСТВИЕ: RAISE\nСУММА: 50\n")

_ACT = (("FOLD", ("FOLD", "ФОЛД", "ПАС", "СБРОС")), ("CHECK", ("CHECK", "ЧЕК", "ПРОПУСК")),
        ("CALL", ("CALL", "КОЛЛ", "УРАВН")), ("RAISE", ("RAISE", "РЕЙЗ", "БЕТ", "BET", "ОЛЛ", "ALL-IN", "ALLIN")))


def parse_action(a):
    s = str(a or "").upper()
    word = "FOLD"
    for canon, keys in _ACT:
        if any(k in s for k in keys):
            word = canon; break
    nums = re.findall(r"\d[\d\s]*", s)
    return word, (re.sub(r"\s", "", nums[0]) if nums else "0")


def fix_response(text):
    """Постобработка: клиент всегда получает ровно три строки в каноническом виде."""
    t = text or ""
    for tok in ("<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>", "<pad>",
                "<|im_end|>", "<|im_start|>", "<end_of_turn>"):
        t = t.replace(tok, "\n")
    t = re.sub(r"^\s*(model|assistant|user|system)\s*$", "", t, flags=re.M).strip()
    a = re.search(r"АНАЛИЗ\s*:\s*([^\n]+)", t, re.I)
    d = re.search(r"ДЕЙСТВИЕ\s*:\s*([^\n]+)", t, re.I)
    s = re.search(r"СУММА\s*:\s*\$?\s*(\d[\d\s]*)", t, re.I)
    analysis = (a.group(1).strip() if a else "Не уверен, играю осторожно.")[:220]
    if not d:
        stats["bad_format"] += 1
        return "АНАЛИЗ: Не уверен, играю осторожно.\nДЕЙСТВИЕ: FOLD\nСУММА: 0", "FOLD", 0, analysis
    act, amt = parse_action(d.group(1))
    amount = int(re.sub(r"\s", "", s.group(1))) if s else (int(amt) if amt.isdigit() else 0)
    if act != "RAISE":
        amount = 0
    return f"АНАЛИЗ: {analysis}\nДЕЙСТВИЕ: {act}\nСУММА: {amount}", act, amount, analysis


def build_body(user_text, stage):
    """ЕДИНАЯ сборка промпта — и для инференса, и для обучения."""
    body = (user_text or "").strip()
    if not body.startswith("[СТАДИЯ:"):
        body = f"[СТАДИЯ: {stage.upper()}]\n{body}"
    if USE_INSTR:
        body += "\n\n" + STAGE_INSTRUCTIONS.get(stage, "") + FEW_SHOT
    return body


def render_prompt(body):
    if USE_CHAT:
        return tokenizer.apply_chat_template([{"role": "user", "content": body}],
                                             tokenize=False, add_generation_prompt=True)
    return body


def ensure_bos(text):
    bos = getattr(tokenizer, "bos_token", None)
    return bos + text if (bos and not text.startswith(bos)) else text


def encode_prompt(text, limit):
    """BOS остаётся первым; при переполнении режем середину, хвост с форматом сохраняем."""
    ids = tokenizer(ensure_bos(text), add_special_tokens=False)["input_ids"]
    if limit and len(ids) > limit:
        stats["truncated"] += 1
        head = ids[:1] if getattr(tokenizer, "bos_token", None) else []
        ids = head + ids[len(ids) - (limit - len(head)):]
    return ids


def build_train_prompt(d):
    stage = str(d.get("stage", "flop")).lower()
    extra = ""
    if d.get("min_raise") is not None: extra += f"\nМИН. РЕЙЗ ДО: ${d['min_raise']}"
    if d.get("max_bet") is not None:   extra += f"\nМАКС. СТАВКА: ${d['max_bet']}"
    if d.get("position"):              extra += f"\nПОЗИЦИЯ: {d['position']}"
    if d.get("players"):               extra += f"\nИГРОКОВ В РАЗДАЧЕ: {d['players']}"
    if d.get("history"):               extra += f"\nДЕЙСТВИЯ НА УЛИЦЕ: {d['history']}"
    return (f"[СТАДИЯ: {stage.upper()}]\nСТАДИЯ: {stage.upper()}\n"
            f"ТВОИ КАРТЫ: {d.get('my_cards','?')}\nБОРД: {d.get('board','—')}\n"
            f"БАНК: ${d.get('pot',0)}, СТЕК: ${d.get('stack',0)}\n"
            f"НУЖНО КОЛЛИРОВАТЬ: ${d.get('to_call',0)}{extra}\n\n"
            f"Ответь СТРОГО:\nАНАЛИЗ: [короткое объяснение]\n"
            f"ДЕЙСТВИЕ: [FOLD/CHECK/CALL/RAISE]\nСУММА: [число]")


# ============================== МОДЕЛЬ ==============================
def adapter_matches(path):
    cfg = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg):
        return True, "нет adapter_config.json"
    try:
        base = json.load(open(cfg, encoding="utf-8")).get("base_model_name_or_path", "")
        norm = lambda s: str(s).lower().rstrip("/").split("/")[-1]
        return (norm(base) == norm(MODEL_ID)), base
    except Exception as e:
        return True, f"не читается: {e}"


def load_model():
    """Загрузка 1-в-1 как в рабочем сервере: NF4 + bf16 + device_map={'':0}. Без autocast/prepare_model."""
    global model, tokenizer, LORA_AVAILABLE, QUANT_ACTIVE, MODEL_DTYPE
    global STOP_IDS, BAD_IDS, END_TURN
    cuda = torch.cuda.is_available()
    MODEL_DTYPE = torch.bfloat16 if cuda else torch.float32

    print("=" * 66)
    print(f"  ЗАГРУЗКА: {MODEL_ID}")
    print(f"  cuda={cuda} | dtype={MODEL_DTYPE} | quant={QUANT} | chat={USE_CHAT} | instr={USE_INSTR}")
    if cuda:
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name} {p.total_memory/1024**3:.1f} ГиБ (sm_{p.major}{p.minor})")
        if VRAM_GB > 0:
            frac = min(0.95, VRAM_GB / (p.total_memory / 1024**3))
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            print(f"  🔒 Лимит VRAM: {VRAM_GB} ГиБ (задан вручную)")
    print("=" * 66)
    if not HF_TOKEN and IS_GEMMA:
        print("❗ HF_TOKEN не задан, а Gemma — gated-репозиторий. Добавь HF_TOKEN в .env")

    tk = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **tk)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    bnb = None
    q = QUANT if cuda else "none"
    if q in ("4bit", "8bit") and not BNB_OK:
        print("⚠️ bitsandbytes недоступен — квантование отключено"); q = "none"
    if q == "4bit":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=MODEL_DTYPE, bnb_4bit_use_double_quant=True)
    elif q == "8bit":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
    QUANT_ACTIVE = bnb is not None

    kw = dict(device_map=({"": 0} if cuda else "cpu"), low_cpu_mem_usage=True, **tk)
    if bnb is not None:
        kw["quantization_config"] = bnb
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=MODEL_DTYPE, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=MODEL_DTYPE, **kw)
    model.eval()

    if PEFT_OK and os.path.exists(LORA_PATH):
        ok, base = adapter_matches(LORA_PATH)
        if not ok and not FORCE_ADAPTER:
            print(f"❌ LoRA в {LORA_PATH} обучена на другой базе: {base} != {MODEL_ID}")
            print("   Адаптер пропущен. Укажи POKER_LORA от этой модели (или POKER_FORCE_ADAPTER=1).")
        else:
            try:
                model = PeftModel.from_pretrained(model, LORA_PATH)
                LORA_AVAILABLE = True
                print(f"✅ LoRA загружена: {LORA_PATH}")
            except Exception as e:
                print(f"⚠️ LoRA не загрузилась: {e}")
    if PEFT_OK and not LORA_AVAILABLE and CREATE_LORA:
        try:
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"])
            model = get_peft_model(model, cfg)
            LORA_AVAILABLE = True
            os.makedirs(LORA_PATH, exist_ok=True)
            print("🆕 Создан новый LoRA-адаптер")
        except Exception as e:
            print(f"⚠️ Не удалось создать LoRA: {e}")

    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name.lower()
    model.eval()

    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    unk = tokenizer.unk_token_id
    END_TURN = "<end_of_turn>" if (eot is not None and eot >= 0 and eot != unk) else None
    ids = [tokenizer.eos_token_id] + ([eot] if END_TURN else [])
    STOP_IDS = sorted({i for i in ids if i is not None and i >= 0})
    BAD_IDS = sorted({i for i in (tokenizer.pad_token_id, tokenizer.unk_token_id)
                      if i is not None and i >= 0})
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = STOP_IDS

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Устройство: {model.device} | обучаемых параметров: {n_train/1e6:.2f}M")
    print(f"🛑 STOP_IDS={STOP_IDS} | BAD_IDS={BAD_IDS} | EOS={tokenizer.eos_token!r} PAD={tokenizer.pad_token!r}")
    print(f"📏 seq={MAX_SEQ_LEN} train_seq={TRAIN_SEQ_LEN} | буфер={MEMORY_LIMIT}")
    print(f"🧠 VRAM: {vram_info()}")
    if n_train == 0:
        print("❗ Обучаемых параметров нет — online-обучение отключено, инференс работает.")
    return n_train


def trainable_params():
    return [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()]


@torch.no_grad()
def run_inference(prompt_text, max_tokens=None, stage_hint=None):
    stage = stage_hint or detect_stage(prompt_text)
    if stage not in ("preflop", "flop", "turn", "river"):
        stage = "flop"
    ids = encode_prompt(render_prompt(build_body(prompt_text, stage)), MAX_SEQ_LEN)
    inp = torch.tensor([ids], dtype=torch.long, device=model.device)
    att = torch.ones_like(inp)
    n_new = min(int(max_tokens or MAX_NEW), 512)

    gen_kw = dict(max_new_tokens=n_new, attention_mask=att,
                  pad_token_id=tokenizer.pad_token_id,
                  eos_token_id=STOP_IDS or tokenizer.eos_token_id,
                  repetition_penalty=REP_PEN)
    if DO_SAMPLE:
        gen_kw.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
    else:
        gen_kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)

    with MODEL_LOCK:                       # без autocast: модель уже в bf16
        out = model.generate(inp, **gen_kw)
    text = tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
    del inp, att, out
    return fix_response(text)


# ============================== ОБУЧЕНИЕ ==============================
def _encode_pair(body, completion):
    """Тот же префикс, что видит инференс. Без двойного BOS."""
    if USE_CHAT:
        pre = tokenizer.apply_chat_template([{"role": "user", "content": body}],
                                            tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}, {"role": "assistant", "content": completion}],
            tokenize=False, add_generation_prompt=False)
    else:
        pre = body + "\n"
        full = pre + completion + (END_TURN or tokenizer.eos_token or "")
    pre_ids = tokenizer(ensure_bos(pre), add_special_tokens=False)["input_ids"]
    ids = tokenizer(ensure_bos(full), add_special_tokens=False)["input_ids"]
    if len(ids) > TRAIN_SEQ_LEN:
        ids = ids[:TRAIN_SEQ_LEN]
    n = 0
    while n < min(len(ids), len(pre_ids)) and ids[n] == pre_ids[n]:
        n += 1
    if n == 0:
        n = min(len(pre_ids), len(ids))
    return ids, n


def train_batch(batch):
    global last_loss
    params = trainable_params()
    if not params:
        return {"status": "error", "error": "нет обучаемых LoRA-параметров"}

    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    opt.zero_grad(set_to_none=True)
    total, valid, skipped = 0.0, 0, 0
    denom = max(1, len(batch))
    print(f"\n🧠 ОБУЧЕНИЕ: {len(batch)} примеров | chat={USE_CHAT} instr={USE_INSTR} | нормировка /{denom}")
    try:
        with MODEL_LOCK:
            model.train()
            for i, item in enumerate(batch):
                try:
                    ids, n_prefix = _encode_pair(item["prompt"], item["completion"])
                    if len(ids) - n_prefix < 4:
                        skipped += 1; continue          # всё замаскировано → НЕ создаём NaN
                    inp = torch.tensor([ids], dtype=torch.long, device=model.device)
                    labels = inp.clone()
                    labels[:, :n_prefix] = -100
                    out = model(input_ids=inp, attention_mask=torch.ones_like(inp), labels=labels)
                    loss = out.loss
                    if not torch.isfinite(loss):
                        skipped += 1; continue
                    (loss * float(item.get("reward", 1.0)) / denom).backward()
                    total += float(loss.item()); valid += 1
                    del out, loss, inp, labels
                except torch.cuda.OutOfMemoryError:
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print(f"⚠️ Пропуск {i}: OOM")
                except Exception as e:
                    skipped += 1
                    print(f"⚠️ Пропуск {i}: {e}")
            if valid:
                norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                if torch.isfinite(norm):
                    opt.step()
                    last_loss = total / valid
                    stats["batches_trained"] += 1
                    threading.Thread(target=save_lora_weights, daemon=True).start()
                    print(f"✅ Loss: {last_loss:.4f} | valid={valid} skip={skipped} | grad_norm={float(norm):.3f}")
                else:
                    stats["train_errors"] += 1
                    print("❌ Нечисловой градиент — шаг пропущен, веса целы")
            else:
                stats["train_errors"] += 1
                print("❌ Нет валидных примеров — шаг пропущен")
            stats["skipped_examples"] += skipped
    finally:
        opt.zero_grad(set_to_none=True)
        model.eval()
        del opt
        free_vram()
    return {"status": "trained" if valid else "error", "loss": last_loss,
            "valid": valid, "skipped": skipped, "buffer_size": len(training_buffer)}


def save_lora_weights():
    try:
        with MODEL_LOCK:
            os.makedirs(LORA_PATH, exist_ok=True)
            model.save_pretrained(LORA_PATH)
            tokenizer.save_pretrained(LORA_PATH)
        print(f"💾 LoRA сохранены: {LORA_PATH}")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения LoRA: {e}")


def trainer_loop():
    """Обучение в фоне — PUT отвечает мгновенно, игра не ждёт."""
    while not STOP_EVENT.is_set():
        TRAIN_EVENT.wait(0.5)
        TRAIN_EVENT.clear()
        if STOP_EVENT.is_set():
            break
        with DATA_LOCK:
            if len(training_buffer) < MEMORY_LIMIT:
                continue
            batch = training_buffer[:MAX_BATCH]
            del training_buffer[:len(batch)]
        save_all_data()
        trainer["running"] = True
        t0 = time.time()
        try:
            res = train_batch(batch)
        except Exception as e:
            traceback.print_exc()
            stats["train_errors"] += 1
            res = {"status": "error", "error": str(e)}
        trainer.update(running=False, last_result=res, reported=False, last_train_ts=time.time())
        print(f"⏱ Батч за {time.time()-t0:.1f}s | VRAM {vram_info()}")
        save_all_data()


def selftest():
    print("\n" + "=" * 66); print("  САМОТЕСТ"); print("=" * 66)
    try:
        t0 = time.time()
        txt, act, amt, _ = run_inference(
            "[СТАДИЯ: PREFLOP]\nСТАДИЯ: ПРЕФЛОП\nТВОИ КАРТЫ: A♠ A♥\nБОРД: —\n"
            "БАНК: $30, СТЕК: $970\nНУЖНО КОЛЛИРОВАТЬ: $20\n\n"
            "Ответь СТРОГО:\nАНАЛИЗ: [объяснение]\nДЕЙСТВИЕ: [FOLD/CHECK/CALL/RAISE]\nСУММА: [число]")
        print(txt)
        print(f"⏱ {time.time()-t0:.1f}s → {act} {amt} | VRAM {vram_info()}")
        print("✅ Модель отвечает" if act else "❌ Пустой ответ")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Самотест упал: {e}")
    print("=" * 66 + "\n")


# ============================== HTTP ==============================
class ModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"       # keep-alive
    timeout = 300
    server_version = "DeepPoker/5.0"

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "86400")

    def _json(self, code, obj, extra=None):
        data = jdump(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/stats", "/health"):
            return self._json(200, {
                "status": "ok", "model": MODEL_ID,
                "lora_available": LORA_AVAILABLE, "lora_active": LORA_AVAILABLE,
                "quantized": QUANT_ACTIVE, "buffer_size": len(training_buffer),
                "buffer_limit": MEMORY_LIMIT, "last_loss": last_loss,
                "device": str(model.device) if model else "—", "dtype": str(MODEL_DTYPE),
                "chat_template": USE_CHAT, "instructions": USE_INSTR,
                "training": trainer["running"], "train_losses": TRAIN_LOSSES,
                "vram": vram_info(), "pretrain_status": pretrain_status,
                "uptime_s": int(time.time() - trainer["started"]), "stats": stats})
        if path == "/buffer":
            return self._json(200, {"buffer_size": len(training_buffer), "examples": training_buffer[:5]})
        if path == "/memory":
            return self._json(200, {"memory": memory_cache[-50:]})
        if path == "/save":
            save_all_data(); save_lora_weights()
            return self._json(200, {"status": "saved", "buffer_size": len(training_buffer)})
        if path == "/selftest":
            try:
                txt, act, amt, _ = run_inference(
                    "СТАДИЯ: FLOP\nТВОИ КАРТЫ: A♠ K♥\nБОРД: A♦ 7♣ 2♠\n"
                    "БАНК: $120, СТЕК: $880\nНУЖНО КОЛЛИРОВАТЬ: $40")
                return self._json(200, {"response": txt, "action": act, "amount": amt})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/pretrain":
            return self.handle_pretrain()
        return self._json(404, {"error": "not found"})

    def handle_pretrain(self):
        global pretrain_status
        if pretrain_status["running"]:
            return self._json(200, {"status": "already_running"})
        try:
            from trainer import generate_training_examples
        except Exception as e:
            return self._json(501, {"error": f"trainer.generate_training_examples недоступен: {e}"})

        def run():
            global pretrain_status
            try:
                pretrain_status = {"running": True, "progress": 10, "message": "Генерация примеров..."}
                ex = generate_training_examples(1000)
                stats["pretrain_examples"] += len(ex)
                with DATA_LOCK:
                    training_buffer.extend(ex)
                save_all_data()
                TRAIN_EVENT.set()
                pretrain_status = {"running": False, "progress": 100,
                                   "message": f"Добавлено {len(ex)} примеров, обучение в фоне"}
            except Exception as e:
                pretrain_status = {"running": False, "progress": -1, "message": f"Ошибка: {e}"}
        threading.Thread(target=run, daemon=True).start()
        return self._json(200, {"status": "started"})

    def do_POST(self):
        try:
            body = self._body()
            prompt = body.get("prompt", "")
            if not prompt:
                return self._json(400, {"error": "empty prompt"})
            if trainer["running"]:
                stats["rejected_503"] += 1
                return self._json(503, {"error": "training in progress", "response": ""},
                                  {"Retry-After": "10"})
            stats["post_requests"] += 1
            stage = detect_stage(prompt)
            t0 = time.time()
            text, act, amount, analysis = run_inference(prompt, body.get("max_tokens", MAX_NEW), stage)
            dt = time.time() - t0
            stats["preflop_queries" if stage == "preflop" else "postflop_queries"] += 1
            print(f"📥 {stage} | {len(prompt)} симв → {act} {amount} за {dt:.1f}s")
            print(f"📤 {text.splitlines()[0][:220]}")
            return self._json(200, {"response": text, "action": act, "amount": amount,
                                    "analysis": analysis, "stage": stage, "elapsed": round(dt, 2)})
        except Exception as e:
            traceback.print_exc()
            return self._json(500, {"error": str(e)})

    def do_PUT(self):
        try:
            body = self._body()
            if not body:
                return self._json(400, {"error": "empty body"})
            hand = {k: body.get(k, d) for k, d in (
                ("stage", "?"), ("my_cards", "?"), ("board", "?"), ("pot", 0), ("to_call", 0),
                ("stack", 0), ("action_taken", "?"), ("outcome", "?"), ("lesson", ""))}
            for k in ("min_raise", "max_bet", "position", "players", "history"):
                if body.get(k) is not None:
                    hand[k] = body[k]
            with DATA_LOCK:
                memory_cache.append(hand)
                del memory_cache[:-1000]
            stats["online_examples"] += 1
            stats["put_requests"] += 1

            won = "WIN" in str(hand["outcome"]).upper()
            stage = str(hand["stage"]).lower()
            if stage not in ("preflop", "flop", "turn", "river"):
                stage = detect_stage(build_train_prompt(hand))
            if won or TRAIN_LOSSES:
                act, amount = parse_action(hand["action_taken"])
                completion = (f"АНАЛИЗ: {hand['lesson'] or 'Играю по ситуации.'}\n"
                              f"ДЕЙСТВИЕ: {act}\nСУММА: {amount}")
                with DATA_LOCK:
                    training_buffer.append({"prompt": build_body(build_train_prompt(hand), stage),
                                            "completion": completion,
                                            "reward": 1.0 if won else 0.2, "stage": stage})
                    size = len(training_buffer)
            else:
                stats["skipped_losses"] += 1
                size = len(training_buffer)
            save_all_data()
            print(f"📝 Буфер: {size}/{MEMORY_LIMIT} ({'WIN' if won else 'LOSS'})")
            if size >= MEMORY_LIMIT:
                TRAIN_EVENT.set()

            resp = {"status": "waiting", "buffer_size": size, "training": trainer["running"]}
            if not trainer["reported"] and trainer["last_result"]:
                r = trainer["last_result"]; trainer["reported"] = True
                resp["status"] = r.get("status", "trained")
                resp["loss"] = r.get("loss", last_loss)
            elif trainer["running"]:
                resp["status"] = "training"
            return self._json(200, resp)
        except Exception as e:
            traceback.print_exc()
            return self._json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        if args and "OPTIONS" in str(args[0]):
            return
        print("🌐 " + (fmt % args if args else fmt))


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve():
    srv = ThreadedHTTPServer((HOST, PORT), ModelHandler)
    threading.Thread(target=trainer_loop, daemon=True, name="Trainer").start()
    print("\n" + "=" * 66)
    print("🚀 СЕРВЕР ЗАПУЩЕН")
    print("=" * 66)
    print(f"📡 http://{HOST}:{PORT}   (в HTML укажи IP этой машины, если играешь с другого устройства)")
    print(f"📊 /stats  💾 /save  📥 /buffer  🧪 /selftest  🎓 /pretrain")
    print(f"🧠 {MODEL_ID} | LoRA: {'✅' if LORA_AVAILABLE else '⚠️ нет'} | VRAM {vram_info()}")
    print("Нажми Ctrl+C для остановки — всё сохранится.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Остановка...")
    finally:
        STOP_EVENT.set(); TRAIN_EVENT.set()
        save_all_data()
        if model is not None and LORA_AVAILABLE:
            save_lora_weights()
        srv.server_close()
        print("✅ Остановлен, данные сохранены")


def main():
    print("\n📂 Загрузка сохранённых данных...")
    load_all_data()
    load_model()
    if SELFTEST:
        selftest()
    serve()


if __name__ == "__main__":
    main()