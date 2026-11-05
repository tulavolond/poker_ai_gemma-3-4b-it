"""
Минимальный клиент покера для теста сервера.
Запуск: python poker_client.py
"""
import json
import random
import time
import urllib.request
from itertools import combinations
from collections import Counter
from dataclasses import dataclass

SERVER_URL = "http://192.168.1.103:8899"


# ============================================================
# КАРТЫ
# ============================================================

@dataclass
class Card:
    rank: int
    suit: str

    def __str__(self):
        names = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        return f"{names.get(self.rank, str(self.rank))}{self.suit}"


class Deck:
    def __init__(self):
        self.cards = [
            Card(r, s) for r in range(2, 15)
            for s in ['♠', '♥', '♦', '♣']
        ]
        random.shuffle(self.cards)

    def deal(self, n=1):
        return [self.cards.pop() for _ in range(n)]


# ============================================================
# ОЦЕНЩИК
# ============================================================

class HandEvaluator:
    CATEGORY_NAMES = {
        8: "Роял-флеш", 7: "Стрит-флеш", 6: "Каре",
        5: "Фулл-хаус", 4: "Флеш", 3: "Стрит",
        2: "Сет", 1: "Две пары", 0: "Пара", -1: "Старшая",
    }

    @staticmethod
    def evaluate(cards):
        best = None
        for combo in combinations(cards, 5):
            score = HandEvaluator._score5(list(combo))
            if best is None or score > best:
                best = score
        return best

    @staticmethod
    def _score5(cards):
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        is_flush = len(set(suits)) == 1

        uniq = sorted(set(ranks), reverse=True)
        is_str = False
        sh = 0
        if len(uniq) == 5:
            if uniq[0] - uniq[4] == 4:
                is_str = True; sh = uniq[0]
            elif uniq == [14, 5, 4, 3, 2]:
                is_str = True; sh = 5

        cnts = sorted(Counter(ranks).values(), reverse=True)

        if is_flush and is_str:
            cat = 8 if sh == 14 else 7
        elif cnts[0] == 4: cat = 6
        elif cnts[0] == 3 and cnts[1] == 2: cat = 5
        elif is_flush: cat = 4
        elif is_str: cat = 3
        elif cnts[0] == 3: cat = 2
        elif cnts[0] == 2 and cnts[1] == 2: cat = 1
        elif cnts[0] == 2: cat = 0
        else: cat = -1

        groups = {}
        for r, c in Counter(ranks).items():
            groups.setdefault(c, []).append(r)
        kickers = []
        for c in sorted(groups.keys(), reverse=True):
            kickers.extend(sorted(groups[c], reverse=True))

        return (cat, tuple(kickers))

    @staticmethod
    def name(score):
        return HandEvaluator.CATEGORY_NAMES.get(score[0], "?")

    @staticmethod
    def preflop(cards):
        r1, r2 = sorted([cards[0].rank, cards[1].rank], reverse=True)
        suited = cards[0].suit == cards[1].suit
        if r1 == r2:
            return 0.5 + (r1 - 2) / 24.0
        score = (r1 + r2) / 28.0 * 0.4
        if suited: score += 0.1
        if r1 == 14: score += 0.1
        if r1 - r2 <= 2: score += 0.05
        return min(max(score, 0), 1.0)


# ============================================================
# ЗАПРОС К СЕРВЕРУ
# ============================================================

def ask_server(prompt, max_tokens=80):
    """Отправляет запрос на сервер и получает ответ."""
    try:
        data = json.dumps({
            'prompt': prompt,
            'max_tokens': max_tokens,
        }).encode('utf-8')

        req = urllib.request.Request(
            SERVER_URL,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode('utf-8'))

        return result.get('response', '')

    except Exception as e:
        print(f"  ⚠️ Ошибка сервера: {e}")
        return ""


def parse_response(response):
    """Парсит ответ модели: ДЕЙСТВИЕ: X | СУММА: N"""
    upper = response.upper()

    if "ФОЛД" in upper or "FOLD" in upper:
        action = "ФОЛД"
    elif "РЕЙЗ" in upper or "RAISE" in upper:
        action = "РЕЙЗ"
    elif "ЧЕК" in upper or "CHECK" in upper:
        action = "ЧЕК"
    elif "КОЛЛ" in upper or "CALL" in upper:
        action = "КОЛЛ"
    else:
        action = "ФОЛД"

    import re
    nums = re.findall(r'\d+', response)
    amount = int(nums[-1]) if nums else 0

    return action, amount


# ============================================================
# ИГРОКИ
# ============================================================

class AIPlayer:
    def __init__(self, name, stack, style, persona):
        self.name = name
        self.stack = stack
        self.style = style
        self.persona = persona
        self.cards = []
        self.current_bet = 0
        self.folded = False

    def reset(self):
        self.cards = []
        self.current_bet = 0
        self.folded = False

    def decide(self, community, stage, pot, to_call, min_raise):
        cards_str = " ".join(str(c) for c in self.cards)
        comm_str = " ".join(str(c) for c in community) if community else "нет"

        if community:
            score = HandEvaluator.evaluate(self.cards + community)
            hand = HandEvaluator.name(score)
            strength = (score[0] + 1) / 9.0
        else:
            hand = "префлоп"
            strength = HandEvaluator.preflop(self.cards)

        prompt = f"""{self.persona}

Текущая ситуация:
- Стадия: {stage}
- Твои карты: {cards_str}
- На столе: {comm_str}
- Твоя рука: {hand}
- Сила руки: {strength:.2f}
- Банк: ${pot}
- Нужно уравнять: ${to_call}
- Минимальный рейз: ${min_raise}
- Твой стек: ${self.stack}

Ответь СТРОГО одной строкой в формате:
ДЕЙСТВИЕ: [ФОЛД/КОЛЛ/ЧЕК/РЕЙЗ] | СУММА: [число]"""

        start = time.time()
        response = ask_server(prompt)
        elapsed = time.time() - start

        action, amount = parse_response(response)
        return action, amount, elapsed, response


# ============================================================
# ГЛАВНЫЙ ЦИКЛ
# ============================================================

def main():
    print("=" * 60)
    print("  ПОКЕР: ТЕСТ СЕРВЕРА")
    print("=" * 60)

    styles = [
        ("тайтовый", "Ты — ТАЙТОВЫЙ игрок. ФОЛД со слабой рукой, РЕЙЗ только с очень сильной (0.7+)."),
        ("агрессивный", "Ты — АГРЕССИВНЫЙ игрок. Часто делай РЕЙЗ даже со средней рукой."),
        ("лузовый", "Ты — ЛУЗОВЫЙ игрок. КОЛЛ почти всегда, ФОЛД только с мусором."),
    ]

    players = [AIPlayer(f"ИИ_{i+1}", 1000, s, p) for i, (s, p) in enumerate(styles)]

    deck = Deck()
    community = []
    pot = 0

    # Раздаём карты
    print("\n📦 Раздача карт...")
    for p in players:
        p.cards = deck.deal(2)
        print(f"  {p.name} [{p.style}]: {p.cards[0]} {p.cards[1]}")

    # Блайнды
    players[0].stack -= 5
    players[1].stack -= 10
    pot = 15
    print(f"\n💰 Блайнды: {players[0].name}=$5, {players[1].name}=$10")
    print(f"🏦 Банк: ${pot}")

    # === ПРЕФЛОП ===
    print("\n" + "=" * 60)
    print("📋 ПРЕФЛОП")
    print("=" * 60)

    current_bet = 10

    for p in players:
        to_call = current_bet - p.current_bet
        action, amount, elapsed, raw = p.decide(
            community, "Префлоп", pot, to_call, 20
        )

        print(f"\n  🎯 {p.name} [{p.style}] ({elapsed:.1f}s):")
        print(f"     {action} ${amount}")
        print(f"     Ответ: {raw[:100]}")

        if action == "РЕЙЗ":
            total = min(amount, p.stack)
            p.stack -= total
            p.current_bet += total
            pot += total
            current_bet = p.current_bet
        elif action == "КОЛЛ":
            actual = min(to_call, p.stack)
            p.stack -= actual
            p.current_bet += actual
            pot += actual
        elif action == "ФОЛД":
            p.folded = True

    # Итоги
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ПРЕФЛОПА")
    print("=" * 60)
    for p in players:
        status = "❌ фолд" if p.folded else "✓"
        print(f"  {status} {p.name}: ${p.stack} (бет ${p.current_bet})")
    print(f"  🏦 Банк: ${pot}")


if __name__ == "__main__":
    main()