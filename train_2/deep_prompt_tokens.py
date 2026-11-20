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

TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("📦 Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
print("✅ Токенизатор загружен\n")

prompt = """[СТАДИЯ: ПРЕФЛОП]

=== ИСТОРИЯ РАЗДАЧИ ===
Hero (BTN) OPEN_RAISE, Villain (BB) CALL

=== ТВОИ КАРТЫ ===
10♣, 8♦

=== АНАЛИЗ СИСТЕМЫ ===
Рука: Слабая связка (сила 30/100)
Диапазон оппонента: AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo,AQo

=== КОНТЕКСТ ===
Позиция: BTN
Стек: $1000, Банк: $15
Колл: $10

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CALL/RAISE]
СУММА: [число]"""

print(f"~250 токенов ~: {len(tokenizer.encode(prompt))}")  # ~250 токенов