# poker_server.py
# python ./dataset_21_11/poker_server.py --adapter qwen25-decisions-lora --port 8000
import argparse
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

HERE = Path(__file__).resolve().parent

LOCK = threading.Lock()
TOK = None
MODEL = None
DEVICE = "cuda"


def find_default_adapter():
    candidates = [
        HERE / "dataset_21_11" / "qwen25-decisions-lora",
        HERE / "qwen25-decisions-lora",
        HERE / "dataset_21_11" / "qwen25-bet-lora",
        HERE / "qwen25-bet-lora",
        HERE / "dataset_21_11" / "qwen25-preflop-lora",
        HERE / "qwen25-preflop-lora",
        HERE / "dataset_21_11" / "qwen25-facing-lora",
        HERE / "qwen25-facing-lora",
    ]

    for path in candidates:
        if path.exists() and (path / "adapter_config.json").exists():
            return str(path)

    return ""


def load_model(base_model_name, adapter_path):
    global TOK, MODEL, DEVICE

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    token = os.getenv("HF_TOKEN")

    adapter_path = Path(adapter_path) if adapter_path else None

    # Токенизатор: если рядом с адаптером есть сохранённый токенизатор, берём его.
    tok_source = base_model_name
    if adapter_path and adapter_path.exists() and (adapter_path / "tokenizer_config.json").exists():
        tok_source = str(adapter_path)

    print(f"[LOAD] tokenizer from: {tok_source}")
    TOK = AutoTokenizer.from_pretrained(tok_source, token=token, trust_remote_code=True)

    if TOK.pad_token is None:
        TOK.pad_token = TOK.eos_token

    print(f"[LOAD] base model: {base_model_name}")
    print(f"[LOAD] device: {DEVICE}, dtype: {dtype}")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=dtype,
            token=token,
            device_map={"": DEVICE},
            trust_remote_code=True,
        )
    except TypeError:
        # Для старых версий transformers
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            token=token,
            device_map={"": DEVICE},
            trust_remote_code=True,
        )

    if adapter_path and adapter_path.exists():
        if PeftModel is None:
            raise RuntimeError("peft is not installed. Run: pip install peft")

        print(f"[LOAD] adapter: {adapter_path}")
        MODEL = PeftModel.from_pretrained(base_model, str(adapter_path)).eval()
    else:
        print("[LOAD] adapter not found, using base model only")
        MODEL = base_model.eval()

    print("[LOAD] done")


def truncate_response(text: str) -> str:
    """
    Обрезаем хвост после первого закрывающего тега ответа.
    Это лечит проблему, когда модель продолжает писать после </action>.
    """
    for tag in ["</action>", "</definition>"]:
        i = text.find(tag)
        if i != -1:
            return text[:i + len(tag)]
    return text


def generate_answer(system: str, user: str) -> str:
    system = (system or "").strip()
    user = (user or "").strip()

    if not user:
        return "ERROR: empty user prompt"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    text = TOK.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = TOK(text, return_tensors="pt").to(DEVICE)

    with LOCK:
        with torch.inference_mode():
            out = MODEL.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=TOK.eos_token_id
            )

    response = TOK.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return truncate_response(response)


def extract_system_user(data: dict):
    """
    Поддерживает несколько форматов:
    1) {"system": "...", "user": "..."}
    2) {"prompt": "[SYSTEM]...[USER]..."}
    3) {"messages": [...]}
    """
    if not isinstance(data, dict):
        return "", ""

    # Формат с messages
    if isinstance(data.get("messages"), list):
        system = ""
        user = ""
        for m in data["messages"]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                user = content
        return system, user

    system = data.get("system", "") or ""
    user = data.get("user", "") or ""

    # Совместимость со старым buildPrompt, который склеивал строку:
    # [SYSTEM] ... [USER] ...
    if not user:
        prompt = data.get("prompt", "") or ""

        m = re.search(
            r"\[SYSTEM\]\s*([\s\S]*?)\s*\[USER\]\s*([\s\S]*)",
            prompt,
            re.IGNORECASE
        )

        if m:
            system = m.group(1).strip()
            user = m.group(2).strip()
        else:
            user = prompt.strip()

    return system, user


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Чтобы не заспамить консоль
        pass

    def _send_bytes(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: dict, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send_bytes(code, "application/json; charset=utf-8", body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"ok": True})
            return

        if self.path in ["/", "/index.html", "/poker_client.html"]:
            html_path = HERE / "poker_client.html"
            if html_path.exists():
                self._send_bytes(
                    200,
                    "text/html; charset=utf-8",
                    html_path.read_bytes()
                )
            else:
                self._send_json(
                    {"error": "poker_client.html not found near poker_server.py"},
                    404
                )
            return

        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/decide":
            self._send_json({"error": "use POST /decide"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send_json({"error": f"bad json: {e}"}, 400)
            return

        system, user = extract_system_user(data)

        if not user:
            self._send_json({"error": "empty user prompt"}, 400)
            return

        try:
            response = generate_answer(system, user)
            self._send_json({"response": response})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    adapter = args.adapter.strip()
    if not adapter:
        adapter = find_default_adapter()
        if adapter:
            print(f"[INFO] auto-detected adapter: {adapter}")
        else:
            print("[INFO] no adapter found, will use base model")

    load_model(args.base, adapter)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print()
    print("=" * 70)
    print(f"Open: http://127.0.0.1:{args.port}")
    print("=" * 70)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()