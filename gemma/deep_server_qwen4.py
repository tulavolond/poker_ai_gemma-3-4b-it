"""
deep_server_v3.py — покерный нейро-сервер (Qwen2.5 + LoRA) + online-дообучение.
Протокол совместим с deep_poker.html:  POST /  |  GET /stats  |  PUT /
Запуск:  python deep_server_v3.py
"""
import os, re, json, time, threading, traceback
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
    PEFT_OK = True
except Exception:
    PeftModel, PEFT_OK = None, False
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ================= КОНФИГ =================
BASE_MODEL_ID    = os.getenv("POKER_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LORA_PATH        = os.getenv("POKER_LORA", "./train_2/dataset_2000_v2")
LORA_ONLINE_PATH = os.getenv("POKER_LORA_OUT", "./train_2/lora_online_learning_qwen")
DATA_DIR         = os.getenv("POKER_DATA_DIR", "./train_2")
MEMORY_FILE = os.path.join(DATA_DIR, "poker_memory.json")
BUFFER_FILE = os.path.join(DATA_DIR, "training_buffer.json")
STATS_FILE  = os.path.join(DATA_DIR, "training_stats.json")
PORT         = int(os.getenv("POKER_PORT", "8899"))
HOST         = os.getenv("POKER_HOST", "0.0.0.0")
MEMORY_LIMIT = 100
MAX_BATCH    = 100
MAX_SEQ_LEN  = 788
MAX_NEW      = 150
LR           = 1e-5
USE_CHAT_TEMPLATE = False   # ЕДИНЫЙ формат для инференса и обучения
TRAIN_LOSSES      = False   # не подкреплять проигравшие действия
GRAD_ACCUM_NORM   = True

os.makedirs(DATA_DIR, exist_ok=True)

# ================= СОСТОЯНИЕ =================
model = None
tokenizer = None
LORA_AVAILABLE = False
MODEL_DTYPE = None
AMP_DTYPE = None
training_buffer, memory_cache = [], []
last_loss = 0.0
stats = {"pretrain_examples": 0, "online_examples": 0, "batches_trained": 0,
         "preflop_queries": 0, "postflop_queries": 0, "post_requests": 0,
         "put_requests": 0, "skipped_losses": 0, "train_errors": 0,
         "bad_format": 0, "rejected_503": 0}
DATA_LOCK   = threading.Lock()
MODEL_LOCK  = threading.RLock()
TRAIN_EVENT = threading.Event()
STOP_EVENT  = threading.Event()
trainer = {"running": False, "last_result": None, "reported": True,
           "started": time.time(), "last_train_ts": 0.0}

# ================= PERSISTENCE =================
def _finite(o):
    if isinstance(o, float):
        return 0.0 if (o != o or o in (float("inf"), float("-inf"))) else round(o, 6)
    if isinstance(o, dict):  return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):  return [_finite(v) for v in o]
    return o

def jdump(obj):
    return json.dumps(_finite(obj), ensure_ascii=False, allow_nan=False)

def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_finite(obj), f, ensure_ascii=False, indent=1, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def save_all_data():
    with DATA_LOCK:
        try:
            _atomic_write(MEMORY_FILE, memory_cache[-100:])
            _atomic_write(BUFFER_FILE, training_buffer)
            _atomic_write(STATS_FILE, stats)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения: {e}")

def load_all_data():
    global training_buffer, memory_cache
    for path, target in ((MEMORY_FILE, "memory"), (BUFFER_FILE, "buffer"), (STATS_FILE, "stats")):
        if not os.path.exists(path): continue
        try:
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            if target == "memory" and isinstance(data, list):
                memory_cache = data; print(f"📂 Память: {len(data)}")
            elif target == "buffer" and isinstance(data, list):
                training_buffer = [x for x in data if isinstance(x, dict) and "prompt" in x]
                print(f"📂 Буфер: {len(training_buffer)}")
            elif target == "stats" and isinstance(data, dict):
                stats.update(data); print("📂 Статистика загружена")
        except Exception as e:
            print(f"⚠️ Не читается {path}: {e}")

# ================= ДЕТЕКТОР УЛИЦЫ =================
_STAGE_MAP = {"PREFLOP": "preflop", "PRE-FLOP": "preflop", "PRE FLOP": "preflop",
              "ПРЕФЛОП": "preflop", "ПРЕ-ФЛОП": "preflop",
              "FLOP": "flop", "ФЛОП": "flop",
              "TURN": "turn", "ТЁРН": "turn", "ТЕРН": "turn",
              "RIVER": "river", "РИВЕР": "river"}

def detect_stage(prompt_text):
    p = (prompt_text or "").upper()
    m = re.search(r"\[\s*СТАДИЯ\s*:\s*([A-ZА-ЯЁ\- ]+?)\s*\]", p)
    if m and m.group(1).strip() in _STAGE_MAP:
        return _STAGE_MAP[m.group(1).strip()]
    m = re.search(r"СТАДИЯ\s*:\s*([A-ZА-ЯЁ\- ]+)", p)
    if m:
        key = m.group(1).strip()
        for k, v in _STAGE_MAP.items():
            if key.startswith(k): return v
    for group in (("ПРЕФЛОП", "PRE-FLOP", "PRE FLOP", "PREFLOP", "ПРЕ-ФЛОП"),
                  ("РИВЕР", "RIVER"), ("ТЁРН", "ТЕРН", "TURN"), ("ФЛОП", "FLOP")):
        if any(k in p for k in group):
            return _STAGE_MAP[group[0]]
    return "unknown"

# ================= ПОСТОБРАБОТКА =================
_ACT_MAP = (("FOLD", ("FOLD", "ФОЛД", "ПАС", "СБРОС")),
            ("CHECK", ("CHECK", "ЧЕК", "ПРОПУСК")),
            ("CALL", ("CALL", "КОЛЛ", "УРАВН")),
            ("RAISE", ("RAISE", "РЕЙЗ", "БЕТ", "BET", "ОЛЛ", "ALL-IN", "ALLIN")))

def parse_action(action):
    s = str(action or "").upper()
    word = "FOLD"
    for canon, keys in _ACT_MAP:
        if any(k in s for k in keys):
            word = canon; break
    nums = re.findall(r"\d[\d\s]*", s)
    amount = re.sub(r"\s", "", nums[0]) if nums else "0"
    return word, amount

def fix_response(text):
    t = text or ""
    for tok in ("<|im_end|>", "<|im_start|>", "<end_of_turn>"):
        t = t.replace(tok, "\n")
    t = re.sub(r"^\s*(assistant|user|system)\s*$", "", t, flags=re.M).strip()
    a = re.search(r"АНАЛИЗ\s*:\s*([^\n]+)", t, re.I)
    d = re.search(r"ДЕЙСТВИЕ\s*:\s*([^\n]+)", t, re.I)
    s = re.search(r"СУММА\s*:\s*\$?\s*(\d[\d\s]*)", t, re.I)
    analysis = (a.group(1).strip() if a else "Не уверен, играю осторожно.")[:220]
    if not d:
        stats["bad_format"] += 1
        return "АНАЛИЗ: Не уверен, играю осторожно.\nДЕЙСТВИЕ: FOLD\nСУММА: 0", "FOLD", 0, analysis
    act, amt = parse_action(d.group(1))
    amount = int(re.sub(r"\s", "", s.group(1))) if s else (int(amt) if amt.isdigit() else 0)
    if act != "RAISE": amount = 0
    return f"АНАЛИЗ: {analysis}\nДЕЙСТВИЕ: {act}\nСУММА: {amount}", act, amount, analysis

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

# ================= МОДЕЛЬ =================
def load_model():
    global model, tokenizer, LORA_AVAILABLE, MODEL_DTYPE, AMP_DTYPE
    cuda = torch.cuda.is_available()
    if cuda:
        AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        AMP_DTYPE = torch.float32
    MODEL_DTYPE = AMP_DTYPE
    print("=" * 60); print("  ЗАГРУЗКА МОДЕЛИ"); print("=" * 60)
    print(f"📦 {BASE_MODEL_ID} | dtype={MODEL_DTYPE}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    kw = dict(device_map="auto" if cuda else None, trust_remote_code=True, low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=MODEL_DTYPE, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=MODEL_DTYPE, **kw)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    if PEFT_OK and os.path.exists(LORA_PATH):
        try:
            model = PeftModel.from_pretrained(model, LORA_PATH, is_trainable=True)
            model.eval(); LORA_AVAILABLE = True
            print(f"✅ LoRA загружена: {LORA_PATH}")
        except Exception as e:
            print(f"⚠️ Ошибка LoRA: {e}")
    else:
        print(f"⚠️ LoRA не найдена или peft не установлен: {LORA_PATH}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Модель на {model.device} | обучаемых параметров: {n_train/1e6:.2f}M")
    if cuda:
        print(f"   Память GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    if n_train == 0:
        print("❗ Обучаемых параметров нет — online-обучение отключено (инференс работает).")
    return n_train

def trainable_params():
    return [p for n, p in model.named_parameters() if p.requires_grad or "lora_" in n.lower()]

def _autocast():
    if torch.cuda.is_available() and AMP_DTYPE in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    from contextlib import nullcontext
    return nullcontext()

@torch.no_grad()
def run_inference(prompt_text, max_tokens=MAX_NEW, stage_hint=None):
    stage = stage_hint or detect_stage(prompt_text)
    if stage not in ("preflop", "flop", "turn", "river"): stage = "flop"
    prompt = prompt_text.strip()
    if not prompt.startswith("[СТАДИЯ:"):
        prompt = f"[СТАДИЯ: {stage.upper()}]\n{prompt}"
    if USE_CHAT_TEMPLATE:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"          # сохраняем ХВОСТ с форматом ответа
    try:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    finally:
        tokenizer.truncation_side = old_side
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_length = enc["input_ids"].shape[1]
    with MODEL_LOCK, _autocast():
        out = model.generate(**enc, max_new_tokens=min(int(max_tokens or MAX_NEW), MAX_NEW),
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             repetition_penalty=1.05)
    text = tokenizer.decode(out[0][input_length:], skip_special_tokens=True).strip()
    return fix_response(text)

# ================= ОБУЧЕНИЕ =================
def _encode_pair(prompt, completion):
    """Единый формат с инференсом + точная маска префикса."""
    if USE_CHAT_TEMPLATE:
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
            tokenize=False, add_generation_prompt=False)
        pre = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)
        add_special = True
    else:
        pre = prompt + "\n"
        full = pre + completion + (tokenizer.eos_token or "")
        add_special = False
    enc = tokenizer(full, return_tensors="pt", truncation=True,
                    max_length=MAX_SEQ_LEN, add_special_tokens=add_special)
    ids = enc["input_ids"][0].tolist()
    pre_ids = tokenizer(pre, add_special_tokens=add_special)["input_ids"]
    n = 0
    while n < min(len(ids), len(pre_ids)) and ids[n] == pre_ids[n]:
        n += 1
    if n == 0: n = min(len(pre_ids), len(ids))
    return enc, n

def train_batch(batch):
    global last_loss
    params = trainable_params()
    if not params:
        return {"status": "error", "error": "нет обучаемых параметров (LoRA не загружена)"}
    orig_dtypes = [p.dtype for p in params]
    for p in params:
        p.requires_grad_(True)
        p.data = p.data.float()                  # fp32 мастер-веса LoRA
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    total, valid, skipped = 0.0, 0, 0
    denom = max(1, len(batch)) if GRAD_ACCUM_NORM else 1
    print(f"\n🧠 ОБУЧЕНИЕ: {len(batch)} примеров (dtype={AMP_DTYPE}, norm=/{denom})")
    try:
        model.train()
        for i, item in enumerate(batch):
            try:
                enc, n_prefix = _encode_pair(item["prompt"], item["completion"])
                input_ids = enc["input_ids"].to(model.device)
                labels = input_ids.clone()
                labels[:, :n_prefix] = -100
                if int((labels != -100).sum()) < 4:
                    skipped += 1; continue       # нечего учить -> НЕ создаём NaN
                with _autocast():
                    out = model(input_ids=input_ids, labels=labels)
                loss = out.loss
                if not torch.isfinite(loss):
                    skipped += 1; continue
                (loss * float(item.get("reward", 1.0)) / denom).backward()
                total += float(loss.item()); valid += 1
                del out, loss
            except torch.cuda.OutOfMemoryError:
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            except Exception as e:
                skipped += 1
                print(f"⚠️ Пропуск {i}: {e}")
        if valid:
            norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            if torch.isfinite(norm):
                optimizer.step()
                last_loss = total / valid
                stats["batches_trained"] += 1
                threading.Thread(target=save_lora_weights, daemon=True).start()
                print(f"✅ Loss: {last_loss:.4f} | valid={valid} skip={skipped} | grad_norm={float(norm):.3f}")
            else:
                stats["train_errors"] += 1
                print("❌ Нечисловой градиент — шаг пропущен, веса не тронуты")
        else:
            stats["train_errors"] += 1
            print("❌ Нет валидных примеров — шаг пропущен")
    finally:
        optimizer.zero_grad(set_to_none=True)
        for p, dt in zip(params, orig_dtypes):
            p.data = p.data.to(dt)
        model.eval()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"status": "trained" if valid else "error",
            "loss": last_loss, "valid": valid, "skipped": skipped, "buffer_size": len(training_buffer)}

def save_lora_weights():
    try:
        with MODEL_LOCK:
            os.makedirs(LORA_ONLINE_PATH, exist_ok=True)
            model.save_pretrained(LORA_ONLINE_PATH)
            tokenizer.save_pretrained(LORA_ONLINE_PATH)
        print(f"💾 LoRA сохранены: {LORA_ONLINE_PATH}")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения LoRA: {e}")

def trainer_loop():
    while not STOP_EVENT.is_set():
        TRAIN_EVENT.wait(0.5); TRAIN_EVENT.clear()
        if STOP_EVENT.is_set(): break
        with DATA_LOCK:
            if len(training_buffer) < MEMORY_LIMIT: continue
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
        print(f"⏱ Батч обработан за {time.time()-t0:.1f}s")
        save_all_data()

# ================= HTTP =================
class ModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"      # keep-alive
    timeout = 120
    server_version = "DeepPoker/3.0"

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
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n <= 0: return {}
        raw = self.rfile.read(n)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors(); self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/stats", "/health"):
            self._json(200, {
                "status": "ok", "lora_available": LORA_AVAILABLE,
                "buffer_size": len(training_buffer), "buffer_limit": MEMORY_LIMIT,
                "last_loss": last_loss, "device": str(model.device) if model else "—",
                "dtype": str(MODEL_DTYPE), "chat_template": USE_CHAT_TEMPLATE,
                "training": trainer["running"], "train_losses": TRAIN_LOSSES,
                "uptime_s": int(time.time() - trainer["started"]),
                "last_train_ts": trainer["last_train_ts"], "stats": stats})
        elif path == "/memory":
            self._json(200, {"memory": memory_cache[-20:]})
        else:
            self._json(404, {"error": "not found"})

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
            self._json(200, {"response": text, "action": act, "amount": amount,
                             "analysis": analysis, "stage": stage, "elapsed": round(dt, 2)})
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    def do_PUT(self):
        try:
            body = self._body()
            if not body:
                return self._json(400, {"error": "empty body"})
            hand = {k: body.get(k, d) for k, d in (
                ("stage", "?"), ("my_cards", "?"), ("board", "?"), ("pot", 0),
                ("to_call", 0), ("stack", 0), ("action_taken", "?"), ("outcome", "?"), ("lesson", ""))}
            for k in ("min_raise", "max_bet", "position", "players", "history"):
                if body.get(k) is not None: hand[k] = body[k]
            with DATA_LOCK:
                memory_cache.append(hand)
                del memory_cache[:-100]
            stats["online_examples"] += 1
            stats["put_requests"] += 1
            outcome = str(hand["outcome"]).upper()
            won = "WIN" in outcome
            if won or TRAIN_LOSSES:
                act, amount = parse_action(hand["action_taken"])
                completion = f"АНАЛИЗ: {hand['lesson'] or 'Играю по ситуации.'}\nДЕЙСТВИЕ: {act}\nСУММА: {amount}"
                with DATA_LOCK:
                    training_buffer.append({"prompt": build_train_prompt(hand),
                                            "completion": completion,
                                            "reward": 1.0 if won else 0.2,
                                            "stage": str(hand["stage"]).lower()})
                    size = len(training_buffer)
            else:
                stats["skipped_losses"] += 1
                size = len(training_buffer)
            save_all_data()
            print(f"📝 Буфер: {size}/{MEMORY_LIMIT} ({'WIN' if won else 'LOSS'})")
            if size >= MEMORY_LIMIT: TRAIN_EVENT.set()
            resp = {"status": "waiting", "buffer_size": size, "training": trainer["running"]}
            if not trainer["reported"] and trainer["last_result"]:
                r = trainer["last_result"]; trainer["reported"] = True
                resp["status"] = r.get("status", "trained")
                resp["loss"] = r.get("loss", last_loss)
            elif trainer["running"]:
                resp["status"] = "training"
            self._json(200, resp)
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        if args and "OPTIONS" in str(args[0]): return
        print("🌐 " + (fmt % args if args else fmt))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def serve():
    srv = ThreadedHTTPServer((HOST, PORT), ModelHandler)
    threading.Thread(target=trainer_loop, daemon=True, name="Trainer").start()
    print("\n" + "=" * 60); print("🚀 СЕРВЕР ЗАПУЩЕН"); print("=" * 60)
    print(f"📡 http://{HOST}:{PORT}  |  📊 /stats  |  🧠 LoRA: {'✅' if LORA_AVAILABLE else '⚠️ базовая'}")
    print(f"📏 MAX_SEQ_LEN={MAX_SEQ_LEN} | формат: {'chat' if USE_CHAT_TEMPLATE else 'raw'} (единый для train/infer)")
    print(f"🎓 Обучение: буфер {MEMORY_LIMIT}, losses={'да' if TRAIN_LOSSES else 'нет'}, фоновый поток")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Остановка...")
    finally:
        STOP_EVENT.set(); TRAIN_EVENT.set()
        save_all_data()
        if model is not None and LORA_AVAILABLE: save_lora_weights()
        srv.server_close()
        print("✅ Остановлен")

def main():
    print("\n📂 Загрузка сохранённых данных...")
    load_all_data()
    load_model()
    serve()

if __name__ == "__main__":
    main()