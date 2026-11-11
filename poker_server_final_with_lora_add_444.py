"""
Сервер покерного ИИ с АВТОМАТИЧЕСКИМ созданием LoRA и сохранением.
"""
import os
import json
import traceback
import torch
import atexit
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
# КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Автоматическое создание LoRA
# ============================================================

USE_LORA = os.path.exists(LORA_PATH) # True
optimizer = None

if os.path.exists(LORA_PATH):
    print(f"📂 Найден существующий LoRA в {LORA_PATH}. Загружаю...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    print("✅ LoRA загружен из файла.")
else:
    print(f"🆕 LoRA не найден. Создаю новый адаптер в {LORA_PATH}...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    os.makedirs(LORA_PATH, exist_ok=True)
    print("✅ Новый LoRA-адаптер создан.")

# === ИСПРАВЛЕНИЕ: Явно включаем градиенты ТОЛЬКО для LoRA ===
trainable_params = 0
all_params = 0

for name, param in model.named_parameters():
    all_params += param.numel()
    if "lora" in name.lower():
        param.requires_grad = True      # Размораживаем веса адаптера
        trainable_params += param.numel()
    else:
        param.requires_grad = False     # Замораживаем базовую модель (экономит память)

print(f"✅ Оптимизатор готов. Обучаемых параметров: {trainable_params:,} (всего параметров в модели: {all_params:,})")

if trainable_params == 0:
    print("⚠️ ВНИМАНИЕ: Обучаемых параметров 0! Возможно, файл адаптера поврежден.")
    print("💡 Рекомендация: удалите папку ./poker_gemma3_lora и запустите сервер заново, чтобы создать её с нуля.")

# Инициализируем оптимизатор ТОЛЬКО на параметрах, где requires_grad == True
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-5, 
    weight_decay=0.01
)

# ============================================================
# ТОКЕНИЗАТОР И ШАБЛОН
# ============================================================
GEMMA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}{{ '<start_of_turn>model\\n' }}"
    "{% else %}{{ '<start_of_turn>' + message['role'] + '\\n' }}{% endif %}"
    "{{ message['content'] + '<end_of_turn>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
)
if tokenizer.chat_template is None:
    tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
stop_ids = sorted(set([x for x in [tokenizer.eos_token_id, end_of_turn_id] if x is not None]))
bad_ids = sorted(set([x for x in [tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id] if x is not None]))

print("✅ Модель готова.")

# ============================================================
# ПАМЯТЬ И СОХРАНЕНИЕ
# ============================================================
training_buffer = []
MEMORY_LIMIT = 30
last_loss = 0.0

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
    if len(memory) > 100:
        memory = memory[-100:]
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def save_lora_weights():
    """ЯВНО сохраняет веса LoRA на диск"""
    print("\n💾 [save_lora_weights] Начинаю сохранение...")
    try:
        model.save_pretrained(LORA_PATH)
        tokenizer.save_pretrained(LORA_PATH)
        print(f"✅ Веса LoRA сохранены в {LORA_PATH}")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
        traceback.print_exc()

# ============================================================
# ОБУЧЕНИЕ НА БАТЧЕ
# ============================================================
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
    
    # Сохраняем СРАЗУ после обучения
    save_lora_weights()
    print(f"✅ Обучено. Loss: {last_loss:.4f}")
    
    training_buffer = []
    return {"status": "trained", "loss": last_loss, "buffer_size": 0}

# ============================================================
# ИНФЕРЕНС
# ============================================================
def run_inference(prompt_text, max_tokens=120):
    messages = [{"role": "user", "content": prompt_text}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.1, eos_token_id=stop_ids, bad_words_ids=[[i] for i in bad_ids],
        )
    
    gen_ids = outputs[0][input_length:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    for tok in ('<end_of_turn>', '<eos>', '<bos>', '<pad>', '<start_of_turn>', 'model\n', 'user\n'):
        text = text.replace(tok, '')
    return text.strip()

# ============================================================
# HTTP-ОБРАБОТЧИК
# ============================================================
class ModelHandler(BaseHTTPRequestHandler):
    def _set_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200); self._set_cors(); self.end_headers()

    def do_GET(self):
        try:
            memory = load_memory()[-50:]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors(); self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'ok', 
                'lora_active': USE_LORA,
                'buffer_size': len(training_buffer),
                'last_loss': last_loss,
                'recent_memory': memory
            }, ensure_ascii=False).encode('utf-8'))
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

            response_text = run_inference(prompt, body.get('max_tokens', 120))
            if not response_text:
                response_text = "АНАЛИЗ: Ошибка генерации\nДЕЙСТВИЕ: FOLD\nСУММА: 0"

            print(f"\n📥 Запрос от ИИ-игрока. Длина промпта: {len(prompt)}")
            # print(f"\n📥промпт: {prompt}")
            print(f"📤 Ответ модели: {response_text[:600]}")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors(); self.end_headers()
            self.wfile.write(json.dumps({'response': response_text}, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            print(f"❌ Ошибка POST: {e}")
            self._send_error(500, str(e))

    def do_PUT(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(length)
            body = json.loads(raw_body.decode('utf-8'))
            
            save_memory({
                "stage": body.get("stage", "?"),
                "my_cards": body.get("my_cards", "?"),
                "board": body.get("board", "?"),
                "action_taken": body.get("action_taken", "?"),
                "outcome": body.get("outcome", "?"),
                "lesson": body.get("lesson", "")
            })
            
            training_buffer.append({
                "prompt": body.get("prompt", ""),
                "completion": body.get("completion", ""),
                "reward": body.get("reward", 1.0)
            })
            
            train_status = train_batch()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors(); self.end_headers()
            self.wfile.write(json.dumps(train_status, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            print(f"❌ Ошибка PUT: {e}")
            self._send_error(500, str(e))

    def _send_error(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self._set_cors(); self.end_headers()
        self.wfile.write(json.dumps({'error': message}, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        if 'OPTIONS' not in str(args[0]): print(f"🌐 {args[0]}")

# ============================================================
# ЗАПУСК С ЯВНЫМ СОХРАНЕНИЕМ ПРИ ОСТАНОВКЕ
# ============================================================
server = HTTPServer((HOST, PORT), ModelHandler)
print(f"\n🚀 Сервер запущен: http://{HOST}:{PORT}")
print(f"🧠 Память: {MEMORY_FILE}")
print(f"🎯 LoRA: {LORA_PATH}")
print("Нажмите Ctrl+C для остановки.\n")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Остановлен пользователем.")
    # КРИТИЧЕСКИ ВАЖНО: ЯВНО вызываем сохранение ПЕРЕД закрытием
    save_lora_weights()
    server.server_close()
    print("✅ Сервер полностью остановлен. Веса сохранены.")