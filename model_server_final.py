"""
Сервер модели для покера (ИСПРАВЛЕННАЯ ВЕРСИЯ).
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

MODEL_ID = "google/gemma-3-4b-it" # Или gemma-2-4b-it, если это опечатка
PORT = 8899

print("=" * 60)
print("  ЗАГРУЗКА МОДЕЛИ")
print("=" * 60)

# 1. ИСПРАВЛЕНИЕ: Возвращаем bfloat16. Float16 нестабилен для Gemma.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  
    bnb_4bit_use_double_quant=True,
)

print("Загрузка базовой модели...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,             # ИСПРАВЛЕНИЕ: bfloat16
    token=HF_TOKEN,
    low_cpu_mem_usage=True,
)

print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

# 2. ИСПРАВЛЕНИЕ: Gemma строго требует pad_token_id = 0. 
# Не присваивайте ему eos_token (id=1), это ломает attention mask в CUDA.
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0

tokenizer.padding_side = "left"  # Правильно для генерации

print(f"✅ EOS: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"✅ PAD: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

GEMMA_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'assistant' %}{{ '<start_of_turn>model\\n' }}{% else %}{{ '<start_of_turn>' + message['role'] + '\\n' }}{% endif %}{{ message['content'] + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"""

if tokenizer.chat_template is None:
    tokenizer.chat_template = GEMMA_CHAT_TEMPLATE


def generate_response(prompt_text: str, max_tokens: int = 80) -> str:
    messages = [{"role": "user", "content": prompt_text}]

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    # 3. ИСПРАВЛЕНИЕ: Четко задаем stop_ids, избегая конфликтов
    eos_id = tokenizer.eos_token_id
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    
    stop_ids = [eos_id]
    if end_of_turn_id is not None and end_of_turn_id != eos_id:
        stop_ids.append(end_of_turn_id)

    # Очистка кэша CUDA перед генерацией для предотвращения утечек памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_new_tokens=5,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=stop_ids,
            pad_token_id=0,  # 4. ИСПРАВЛЕНИЕ: Явно указываем 0 здесь
        )

    generated_ids = outputs[0][input_length:]

    if len(generated_ids) == 0:
        return ""

    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    response = response.replace('<end_of_turn>', '').replace('<eos>', '').strip()
    print(f"response: {response}")

    return response


class ModelHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode())

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._send_error(400, "Нет тела запроса")
                return

            raw = self.rfile.read(length)
            body = json.loads(raw.decode('utf-8'))

            prompt = body.get('prompt', '')
            max_tokens = body.get('max_tokens', 80)

            if not prompt.strip():
                self._send_error(400, "Пустой промпт")
                return

            response = generate_response(prompt, max_tokens)
            print(f"response2: {response}")
            
            if not response:
                response = "ДЕЙСТВИЕ: ФОЛД | СУММА: 0"

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'response': response,
            }, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            print(f"  ❌ Ошибка сервера: {e}", flush=True)
            traceback.print_exc()
            self._send_error(500, str(e))

    def _send_error(self, code, msg):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': msg}, ensure_ascii=False).encode())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(('0.0.0.0', PORT), ModelHandler)
    print(f"\n🚀 Сервер запущен: http://192.168.1.103:{PORT}")
    print("Нажми Ctrl+C для остановки.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен")
        server.server_close()