"""
Сервер покерного ИИ (Гибридный режим: 1 ИИ на сервере, остальные локально).
Запуск: python model_server.py
"""
import os
import json
import traceback
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from http.server import HTTPServer, BaseHTTPRequestHandler

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "google/gemma-3-4b-it"
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

print("✅ Модель и токенизатор готовы. STOP IDs:", stop_ids)

def run_inference(prompt_text, max_tokens=100):
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

class ModelHandler(BaseHTTPRequestHandler):
    def _set_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200); self._set_cors(); self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self._set_cors(); self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok', 'model': 'gemma-3-4b-it'}).encode('utf-8'))

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._send_error(400, "Empty body")
                return
            
            raw_body = self.rfile.read(length)
            # Безопасный парсинг с защитой от битых запросов
            try:
                body = json.loads(raw_body.decode('utf-8'))
            except json.JSONDecodeError:
                print(f"❌ Ошибка JSON. Сырые данные: {raw_body}")
                self._send_error(400, "Invalid JSON")
                return

            prompt = body.get('prompt', '')
            if not prompt:
                self._send_error(400, "Empty prompt")
                return

            print(f"\n📥 Запрос от ИИ-игрока. Длина промпта: {len(prompt)}")
            response_text = run_inference(prompt, body.get('max_tokens', 100))
            
            # Fallback, если модель всё же вернула пустоту
            if not response_text:
                response_text = "АНАЛИЗ: Ошибка генерации\nДЕЙСТВИЕ: FOLD\nСУММА: 0"

            print(f"📤 Ответ модели: {response_text[:300]}...")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._set_cors(); self.end_headers()
            self.wfile.write(json.dumps({'response': response_text}, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            traceback.print_exc()
            self._send_error(500, str(e))

    def _send_error(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self._set_cors(); self.end_headers()
        self.wfile.write(json.dumps({'error': message}, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        if 'OPTIONS' not in str(args[0]): print(f"🌐 {args[0]}")

server = HTTPServer((HOST, PORT), ModelHandler)
print(f"\n🚀 Сервер запущен: http://{HOST}:{PORT}\n")
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Остановлен")
    server.server_close()