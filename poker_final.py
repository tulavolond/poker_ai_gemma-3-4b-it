"""
ПОКЕР: ЧЕЛОВЕК ПРОТИВ ИИ
Запуск:
  1. python model_server.py
  2. python poker_final.py
"""
import json
import random
import re
import time
import urllib.request
from itertools import combinations
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional
import urllib.error

# ============================================================
# НАСТРОЙКИ
# ============================================================

SERVER_URL = "http://192.168.1.103:8899"

PAUSE_ACTION = 0.4
PAUSE_STAGE = 1.0
PAUSE_CARDS = 0.8

SMALL_BLIND = 5
BIG_BLIND = 10
START_STACK = 1000

MAX_RAISES_PER_ROUND = 3          # FIX: вынес в константу
MAX_ITERATIONS = 10               # FIX: уменьшил


# ============================================================
# 1. КАРТЫ И КОЛОДА
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
        self.cards = []
        self.reset()

    def reset(self):
        self.cards = [
            Card(r, s)
            for r in range(2, 15)
            for s in ['♠', '♥', '♦', '♣']
        ]
        random.shuffle(self.cards)

    def deal(self, n=1):
        return [self.cards.pop() for _ in range(n)]


# ============================================================
# 2. ОЦЕНЩИК РУК
# ============================================================

class HandEvaluator:
    CATEGORY_NAMES = {
        8: "Роял-флеш", 7: "Стрит-флеш", 6: "Каре",
        5: "Фулл-хаус", 4: "Флеш", 3: "Стрит",
        2: "Сет", 1: "Две пары", 0: "Пара", -1: "Старшая карта",
    }

    @staticmethod
    def evaluate(cards: List[Card]) -> Tuple[int, tuple]:
        best = None
        for combo in combinations(cards, 5):
            score = HandEvaluator._score5(list(combo))
            if best is None or score > best:
                best = score
        return best

    @staticmethod
    def _score5(cards: List[Card]) -> Tuple[int, tuple]:
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        is_flush = len(set(suits)) == 1

        is_str = False
        str_high = 0
        uniq = sorted(set(ranks), reverse=True)
        if len(uniq) == 5:
            if uniq[0] - uniq[4] == 4:
                is_str = True
                str_high = uniq[0]
            elif uniq == [14, 5, 4, 3, 2]:
                is_str = True
                str_high = 5

        counts = Counter(ranks)
        cnt_vals = sorted(counts.values(), reverse=True)

        if is_flush and is_str:
            cat = 8 if str_high == 14 else 7
        elif cnt_vals[0] == 4:
            cat = 6
        elif cnt_vals[0] == 3 and cnt_vals[1] == 2:
            cat = 5
        elif is_flush:
            cat = 4
        elif is_str:
            cat = 3
        elif cnt_vals[0] == 3:
            cat = 2
        elif cnt_vals[0] == 2 and cnt_vals[1] == 2:
            cat = 1
        elif cnt_vals[0] == 2:
            cat = 0
        else:
            cat = -1

        groups = {}
        for rank, cnt in counts.items():
            groups.setdefault(cnt, []).append(rank)

        kickers = []
        for cnt in sorted(groups.keys(), reverse=True):
            kickers.extend(sorted(groups[cnt], reverse=True))

        return (cat, tuple(kickers))

    @staticmethod
    def category_name(score: Tuple[int, tuple]) -> str:
        return HandEvaluator.CATEGORY_NAMES.get(score[0], "Неизвестно")

    @staticmethod
    def preflop_strength(cards: List[Card]) -> float:
        if len(cards) < 2:
            return 0.0
        r1, r2 = sorted([cards[0].rank, cards[1].rank], reverse=True)
        suited = cards[0].suit == cards[1].suit
        score = 0.0
        if r1 == r2:
            return min(0.5 + (r1 - 2) / 24.0, 1.0)
        score += (r1 + r2) / 28.0 * 0.4
        if suited:
            score += 0.1
        if r1 - r2 <= 2:
            score += 0.05
        if r1 == 14:
            score += 0.1
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def postflop_strength(cards: List[Card], community: List[Card]) -> float:
        all_cards = cards + community
        score = HandEvaluator.evaluate(all_cards)
        return (score[0] + 1) / 9.0

    @staticmethod
    def draw_potential(cards: List[Card], community: List[Card]) -> Tuple[float, str]:
        all_cards = cards + community
        if len(all_cards) < 5:
            return 0.0, ""

        bonus = 0.0
        draws = []

        suit_counts = Counter(c.suit for c in all_cards)
        for suit, count in suit_counts.items():
            if count == 4:
                bonus += 0.20
                draws.append("флеш-дро")
                break

        ranks = sorted(set(c.rank for c in all_cards))
        for start in range(2, 11):
            window = set(range(start, start + 5))
            if len(window.intersection(ranks)) == 4:
                bonus += 0.15
                draws.append("стрит-дро")
                break

        return min(bonus, 0.35), ", ".join(draws)

    @staticmethod
    def effective_strength(cards, community, stage):
        if not community:
            strength = HandEvaluator.preflop_strength(cards)
            return strength, "префлоп"

        base = HandEvaluator.postflop_strength(cards, community)
        draw_bonus, draw_desc = HandEvaluator.draw_potential(cards, community)

        if stage == "Ривер":
            draw_bonus = 0.0
            draw_desc = ""

        effective = min(base + draw_bonus, 1.0)

        desc_parts = []
        all_cards = cards + community
        score = HandEvaluator.evaluate(all_cards)
        desc_parts.append(HandEvaluator.category_name(score))
        if draw_desc:
            desc_parts.append(f"+ {draw_desc}")

        return effective, " + ".join(desc_parts)

    @staticmethod
    def strength_desc(strength: float) -> str:
        if strength >= 0.8: return "очень сильная"
        elif strength >= 0.6: return "сильная"
        elif strength >= 0.4: return "средняя"
        elif strength >= 0.2: return "слабая"
        else: return "очень слабая"


# ============================================================
# 3. СВЯЗЬ С СЕРВЕРОМ
# ============================================================

# def ask_server(prompt: str, max_tokens: int = 80) -> str:
#     try:
#         data = json.dumps({
#             'prompt': prompt,
#             'max_tokens': max_tokens,
#         }).encode('utf-8')

#         req = urllib.request.Request(
#             SERVER_URL,
#             data=data,
#             headers={'Content-Type': 'application/json'},
#             method='POST',
#         )

#         with urllib.request.urlopen(req, timeout=120) as resp:
#             result = json.loads(resp.read().decode('utf-8'))

#         return result.get('response', '')

#     except KeyboardInterrupt:
#         # FIX: пробрасываем наверх, чтобы main() корректно завершился
#         raise
#     except Exception as e:
#         print(f"  ⚠️ Ошибка сервера: {e}", flush=True)
#         return ""

def ask_server(prompt: str, max_tokens: int = 80) -> str:
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
            # print(f"response: {result.get('response', '')}")
            return result.get('response', '')

    except urllib.error.HTTPError as e:
        # Читаем реальный текст ошибки, который вернул сервер (например, трассировку CUDA)
        error_body = e.read().decode('utf-8')
        print(f"\n  ❌ КРИТИЧЕСКАЯ ОШИБКА СЕРВЕРА (HTTP {e.code}):", flush=True)
        print(f"  {error_body}\n", flush=True)
        return ""
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  ⚠️ Ошибка сети или сервера: {e}", flush=True)
        return ""

def parse_ai_response(response: str) -> Tuple[str, int]:
    upper = response.upper()

    if "ФОЛД" in upper or "FOLD" in upper:
        action = "ФОЛД"
    elif "РЕЙЗ" in upper or "RAISE" in upper or "ПОВЫШ" in upper:
        action = "РЕЙЗ"
    elif "ЧЕК" in upper or "CHECK" in upper:
        action = "ЧЕК"
    elif "КОЛЛ" in upper or "CALL" in upper:
        action = "КОЛЛ"
    else:
        # FIX: если LLM ответила мусором — не фолдим, а чекаем/коллим.
        # Это безопаснее, чем терять игрока из-за ошибки сервера.
        action = "ЧЕК"

    nums = re.findall(r'\d+', response)
    amount = int(nums[-1]) if nums else 0

    if action in ["ЧЕК", "ФОЛД"]:
        amount = 0

    return action, amount


# ============================================================
# 4. БАЗОВЫЙ ИГРОК
# ============================================================

class BasePlayer:
    def __init__(self, name: str, stack: int):
        self.name = name
        self.stack = stack
        self.cards: List[Card] = []
        self.current_bet = 0
        self.folded = False
        self.eliminated = False

    def reset_hand(self):
        self.cards = []
        self.current_bet = 0
        self.folded = False

    def is_human(self) -> bool:
        return False


# ============================================================
# 5. ЧЕЛОВЕК-ИГРОК
# ============================================================

class HumanPlayer(BasePlayer):
    def __init__(self, name: str = "ТЫ", stack: int = START_STACK):
        super().__init__(name, stack)

    def is_human(self) -> bool:
        return True

    def make_decision(self, game_state: dict) -> Tuple[str, int, str, float]:
        start = time.time()

        community = game_state['community_cards']
        to_call = game_state['to_call']
        pot = game_state['pot']
        stage = game_state['stage']

        strength, desc = HandEvaluator.effective_strength(
            self.cards, community, stage
        )
        strength_word = HandEvaluator.strength_desc(strength)

        print(f"\n  {'─'*60}", flush=True)
        print(f"  🎯 ТВОЙ ХОД  [{stage}]", flush=True)
        print(f"  {'─'*60}", flush=True)
        print(f"  🃏 Твои карты: {self.cards[0]} {self.cards[1]}", flush=True)
        if community:
            comm_str = " ".join(str(c) for c in community)
            print(f"  🎴 На столе: {comm_str}", flush=True)
            print(f"  📊 Твоя рука: {desc}", flush=True)
            print(f"  💪 Сила: {strength:.2f} ({strength_word})", flush=True)
        print(f"  🏦 Банк: ${pot}", flush=True)
        print(f"  💼 Твой стек: ${self.stack}", flush=True)

        if to_call > 0:
            print(f"  📞 Нужно уравнять: ${to_call}", flush=True)
            print(f"\n  [ф]олд | [к]олл ${to_call} | [р]ейз", flush=True)
        else:
            print(f"\n  [ч]ек | [р]ейз", flush=True)

        while True:
            try:
                user_input = input("\n  ➤ ").strip().lower()

                if user_input in ['ф', 'фолд', 'fold', 'f']:
                    if to_call == 0:
                        print("  ⚠️ Нельзя фолдить когда можно чек.", flush=True)
                        continue
                    self.folded = True
                    print(f"\n  👁️  Ты фолдишь. Наблюдаешь за игрой ИИ...",
                          flush=True)
                    return "ФОЛД", 0, "👤 Человек", time.time() - start

                elif user_input in ['ч', 'чек', 'check', 'x']:
                    if to_call > 0:
                        print(f"  ⚠️ Нужно уравнять ${to_call} или фолд.",
                              flush=True)
                        continue
                    return "ЧЕК", 0, "👤 Человек", time.time() - start

                elif user_input in ['к', 'колл', 'call', 'c']:
                    if to_call == 0:
                        print("  ⚠️ Нечего коллировать.", flush=True)
                        continue
                    actual = min(to_call, self.stack)
                    return "КОЛЛ", actual, "👤 Человек", time.time() - start

                elif user_input in ['р', 'рейз', 'raise', 'r']:
                    min_raise = game_state['min_raise']
                    if self.stack <= to_call:
                        return "РЕЙЗ", self.stack, "👤 Человек", time.time() - start

                    print(f"  💰 Минимум: ${min_raise} | Твой стек: ${self.stack}",
                          flush=True)
                    while True:
                        try:
                            amount = int(input("  ➤ Сумма рейза: ").strip())
                            if amount < min_raise:
                                print(f"  ⚠️ Минимум ${min_raise}.", flush=True)
                                continue
                            if amount > self.stack:
                                print(f"  ⚠️ У тебя только ${self.stack}.",
                                      flush=True)
                                continue
                            return "РЕЙЗ", amount, "👤 Человек", time.time() - start
                        except ValueError:
                            print("  ⚠️ Введи число.", flush=True)

                elif user_input in ['а', 'ол', 'allin', 'all-in']:
                    return "РЕЙЗ", self.stack, "👤 Человек", time.time() - start

                else:
                    print("  ⚠️ Введи: [ф]олд | [к]олл | [ч]ек | [р]ейз | [а]л-ин",
                          flush=True)

            except KeyboardInterrupt:
                print("\n  ⚠️ Прервано. ФОЛД.", flush=True)
                self.folded = True
                return "ФОЛД", 0, "👤 Человек", time.time() - start


# ============================================================
# 6. ИИ-ИГРОК
# ============================================================

class AIPlayer(BasePlayer):
    def __init__(self, name: str, stack: int, style: str, persona: str):
        super().__init__(name, stack)
        self.style = style
        self.persona = persona
        self.llm_calls = 0

    def make_decision(self, game_state: dict) -> Tuple[str, int, str, float]:
        start = time.time()

        community = game_state['community_cards']
        stage = game_state['stage']
        to_call = game_state['to_call']
        pot = game_state['pot']

        strength, desc = HandEvaluator.effective_strength(
            self.cards, community, stage
        )
        strength_word = HandEvaluator.strength_desc(strength)

        random_factor = random.uniform(0.9, 1.1)
        adjusted_strength = min(strength * random_factor, 1.0)

        cards_str = " ".join(str(c) for c in self.cards)
        comm_str = " ".join(str(c) for c in community) if community else "нет"

        # FIX: добавил явное напоминание про to_call=0 → ЧЕК
        check_hint = ""
        if to_call == 0:
            check_hint = ("Сейчас ставок нет. Если не хочешь рейзить — "
                          "выбери ЧЕК (не КОЛЛ).")

        prompt = f"""{self.persona}

Текущая ситуация:
- Стадия: {stage}
- Твои карты: {cards_str}
- На столе: {comm_str}
- Твоя рука: {desc}
- Сила руки: {adjusted_strength:.2f} ({strength_word})
- Банк: ${pot}
- Нужно уравнять: ${to_call}
- Минимальный рейз (общая ставка): ${game_state['min_raise']}
- Твой стек: ${self.stack}
{check_hint}

Выбери ОДНО действие: ФОЛД, КОЛЛ, ЧЕК или РЕЙЗ.
Правила по сумме:
- ЧЕК: сумма 0 (можно только если to_call=0)
- ФОЛД: сумма 0
- КОЛЛ: сумма {to_call}
- РЕЙЗ: сумма минимум {game_state['min_raise']} (общая ставка)

Ответь СТРОГО одной строкой:
ДЕЙСТВИЕ: [ФОЛД/КОЛЛ/ЧЕК/РЕЙЗ] | СУММА: [число]"""

        response = ask_server(prompt)
        # print(f"response:: {response}")
        elapsed = time.time() - start
        self.llm_calls += 1

        action, amount = parse_ai_response(response)

        # === ВАЛИДАЦИЯ ===
        if action == "КОЛЛ":
            if to_call == 0:
                # FIX: нельзя коллить когда нечего — превращаем в чек
                action = "ЧЕК"
                amount = 0
            else:
                amount = min(to_call, self.stack)

        elif action == "ЧЕК":
            if to_call > 0:
                # FIX: нельзя чекать когда надо коллить — превращаем в колл
                action = "КОЛЛ"
                amount = min(to_call, self.stack)
            else:
                amount = 0

        elif action == "РЕЙЗ":
            # FIX: корректный потолок рейза. Считаем от текущей ставки,
            # а не от всего банка. Максимум — банк + текущая ставка,
            # но не больше стека.
            min_raise = game_state['min_raise']
            max_reasonable = min(
                pot + to_call + BIG_BLIND * 5,   # не больше банка + запас
                self.stack
            )
            amount = max(amount, min_raise)
            amount = min(amount, max_reasonable)
            if amount <= to_call:
                # Если рейз меньше колла — это просто колл
                action = "КОЛЛ" if to_call > 0 else "ЧЕК"
                amount = min(to_call, self.stack) if to_call > 0 else 0

        elif action == "ФОЛД":
            if to_call == 0:
                # FIX: нельзя фолдить когда можно чек — чекаем
                action = "ЧЕК"
                amount = 0
            else:
                amount = 0

        return action, amount, "🧠 LLM", elapsed


# ============================================================
# 7. ИГРОВОЙ МЕНЕДЖЕР
# ============================================================

class PokerGame:
    def __init__(self, players: List[BasePlayer]):
        self.players = players
        self.deck = Deck()
        self.community_cards: List[Card] = []
        self.pot = 0
        self.current_bet = 0
        self.dealer_pos = 0
        self.hand_start_time = 0

    def alive_count(self) -> int:
        return sum(1 for p in self.players if not p.eliminated)

    def alive_players(self) -> List[BasePlayer]:
        return [p for p in self.players if not p.eliminated]

    def _get_action_order(self, alive: List[BasePlayer]) -> List[BasePlayer]:
        n = len(alive)
        start = (self.dealer_pos + 3) % n
        order = []
        for i in range(n):
            order.append(alive[(start + i) % n])
        return order

    def _print_stacks(self, alive: List[BasePlayer],
                      prev_stacks: dict = None,
                      title: str = "Балансы"):
        print(f"\n  💼 {title}", flush=True)
        print(f"  {'─'*55}", flush=True)
        for p in alive:
            stack_str = f"${p.stack}"
            diff_str = ""
            if prev_stacks and p.name in prev_stacks:
                diff = p.stack - prev_stacks[p.name]
                if diff > 0:
                    diff_str = f"  (+{diff})"
                elif diff < 0:
                    diff_str = f"  ({diff})"

            bar_len = max(0, min(p.stack // 50, 20))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            marker = " 👈" if p.is_human() else ""
            print(f"  {p.name:<12} {bar} {stack_str:<8}{diff_str}{marker}",
                  flush=True)
        print(f"  {'─'*55}", flush=True)

    def check_eliminations(self):
        for p in self.players:
            if not p.eliminated and p.stack <= 0:
                p.eliminated = True
                print(f"\n  💀 {p.name} потерял все фишки и ВЫБЫВАЕТ!",
                      flush=True)
                time.sleep(PAUSE_STAGE)

    def _show_ai_cards(self, alive: List[BasePlayer], stage: str = ""):
        active_ai = [p for p in alive
                     if not p.is_human() and not p.folded and not p.eliminated]

        if not active_ai:
            return

        print(f"\n  👁️  РЕЖИМ НАБЛЮДАТЕЛЯ — карты ИИ открыты:", flush=True)
        for p in active_ai:
            strength, desc = HandEvaluator.effective_strength(
                p.cards, self.community_cards, stage or "Префлоп"
            )
            print(f"     🤖 {p.name}: {p.cards[0]} {p.cards[1]}  "
                  f"({desc}, сила {strength:.2f})", flush=True)
        print(f"  {'─'*55}", flush=True)

    def play_hand(self, hand_number: int) -> Optional[dict]:
        self.hand_start_time = time.time()
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0

        alive = self.alive_players()

        for p in alive:
            p.reset_hand()
            p.cards = self.deck.deal(2)

        n = len(alive)
        if n < 2:
            return None

        sb_pos = (self.dealer_pos + 1) % n
        bb_pos = (self.dealer_pos + 2) % n

        prev_stacks = {p.name: p.stack for p in alive}

        print(f"\n{'╔' + '═'*76 + '╗'}", flush=True)
        print(f"  🎰  РАЗДАЧА #{hand_number}  |  Игроков: {n}", flush=True)
        print(f"{'╚' + '═'*76 + '╝'}", flush=True)

        print(f"\n  📦 Раздача карт...", flush=True)
        time.sleep(PAUSE_CARDS)

        for i in range(0, n, 3):
            parts = []
            for j in range(i, min(i + 3, n)):
                p = alive[j]
                if p.is_human():
                    parts.append(f"👤 {p.name}: {p.cards[0]} {p.cards[1]}")
                else:
                    parts.append(f"🤖 {p.name}: 🂠 🂠 {p.cards[0]} {p.cards[1]}")
            print(f"  {'  |  '.join(parts)}", flush=True)

        time.sleep(PAUSE_CARDS)

        sb_p = alive[sb_pos]
        bb_p = alive[bb_pos]

        self._post_blind(sb_p, SMALL_BLIND)
        self._post_blind(bb_p, BIG_BLIND)
        self.current_bet = BIG_BLIND

        print(f"\n  💰 Блайнды: {sb_p.name}=${SMALL_BLIND}, "
              f"{bb_p.name}=${BIG_BLIND}", flush=True)
        print(f"  🏦 Банк: ${self.pot}", flush=True)

        self._print_stacks(alive, prev_stacks, "Стеки перед раздачей")

        result = {
            'hand_number': hand_number,
            'winner': None,
            'pot': 0,
            'winning_hand': "",
        }

        stages = [
            ('Префлоп', 0),
            ('Флоп', 3),
            ('Тёрн', 1),
            ('Ривер', 1),
        ]

        for stage_name, cards_to_deal in stages:
            if cards_to_deal > 0:
                new_cards = self.deck.deal(cards_to_deal)
                self.community_cards.extend(new_cards)

                time.sleep(PAUSE_STAGE)
                comm_str = " ".join(str(c) for c in self.community_cards)
                print(f"\n  {'─'*76}", flush=True)
                print(f"  🃏 {stage_name.upper()}: {comm_str}", flush=True)
                print(f"  {'─'*76}", flush=True)
                time.sleep(PAUSE_CARDS)
            else:
                time.sleep(PAUSE_STAGE)
                print(f"\n  {'─'*76}", flush=True)
                print(f"  📋 {stage_name.upper()}", flush=True)
                print(f"  {'─'*76}", flush=True)

            if stage_name != 'Префлоп':
                self.current_bet = 0
                for p in alive:
                    p.current_bet = 0

            active = [p for p in alive if not p.folded]
            if len(active) > 1:
                self._betting_round(stage_name, alive)

            active = [p for p in alive if not p.folded]
            if len(active) == 1:
                winner = active[0]
                winner.stack += self.pot
                result['winner'] = winner.name
                result['pot'] = self.pot
                result['winning_hand'] = "Все сбросили"
                self._print_result(result, alive, prev_stacks)
                return result

            print(f"\n  🏦 Банк: ${self.pot}", flush=True)

        # === ВСКРЫТИЕ ===
        time.sleep(PAUSE_STAGE)
        print(f"\n  {'─'*76}", flush=True)
        print(f"  🔓 ВСКРЫТИЕ КАРТ", flush=True)
        print(f"  {'─'*76}", flush=True)

        active = [p for p in alive if not p.folded]
        for p in active:
            all_cards = p.cards + self.community_cards
            score = HandEvaluator.evaluate(all_cards)
            hand_name = HandEvaluator.category_name(score)
            marker = " 👈" if p.is_human() else ""
            print(f"  {p.name}: {p.cards[0]} {p.cards[1]} → {hand_name}{marker}",
                  flush=True)
            time.sleep(0.3)

        winners, winning_hand = self._showdown(alive)

        # === FIX: split pot ===
        if winners:
            share = self.pot // len(winners)
            remainder = self.pot - share * len(winners)
            for i, w in enumerate(winners):
                # остаток отдаём первому (или человеку, если он среди победителей)
                extra = remainder if i == 0 else 0
                w.stack += share + extra

            if len(winners) == 1:
                result['winner'] = winners[0].name
            else:
                result['winner'] = " + ".join(w.name for w in winners) + " (сплит)"

            result['pot'] = self.pot
            result['winning_hand'] = winning_hand

        self._print_result(result, alive, prev_stacks)
        return result

    def _print_result(self, result, alive=None, prev_stacks=None):
        hand_time = time.time() - self.hand_start_time
        print(f"\n  {'═'*76}", flush=True)
        print(f"  🏆 Победитель: {result['winner']}", flush=True)
        print(f"  💰 Банк: ${result['pot']}", flush=True)
        if result['winning_hand']:
            print(f"  🃏 Комбинация: {result['winning_hand']}", flush=True)
        print(f"  ⏱  Время: {hand_time:.1f}s", flush=True)
        print(f"  {'═'*76}", flush=True)

        if alive and prev_stacks:
            self._print_stacks(alive, prev_stacks, "Стеки после раздачи")

    def _post_blind(self, player: BasePlayer, amount: int):
        actual = min(amount, player.stack)
        player.stack -= actual
        player.current_bet = actual
        self.pot += actual

    # ========================================================
    # FIX: полностью переписан _betting_round
    # ========================================================
    def _betting_round(self, stage: str, alive: List[BasePlayer]):
        """
        Раунд торгов.
        Логика:
        - Идём по кругу, пока все не уравняют.
        - Рейзы ограничены MAX_RAISES_PER_ROUND.
        - Когда рейзы кончились, принудительно коллим/чекаем ОДИН раз
          и выходим из раунда, если все уравняли.
        - Жёсткий лимит итераций — защита от зацикливания.
        """
        action_order = self._get_action_order(alive)
        total_raises = 0

        human = next((p for p in alive if p.is_human()), None)
        human_folded = human is not None and human.folded

        if human_folded:
            self._show_ai_cards(alive, stage)

        for iteration in range(MAX_ITERATIONS):
            any_action = False

            for player in action_order:
                if player.folded or player.eliminated or player.stack == 0:
                    continue

                active = [p for p in alive
                          if not p.folded and not p.eliminated]
                if len(active) <= 1:
                    return

                to_call = self.current_bet - player.current_bet
                min_raise = self.current_bet + BIG_BLIND

                # === РЕЙЗЫ КОНЧИЛИСЬ ===
                if total_raises >= MAX_RAISES_PER_ROUND:
                    # Игрок может только колл или чек
                    if to_call > 0:
                        actual = min(to_call, player.stack)
                        if actual > 0:
                            player.stack -= actual
                            player.current_bet += actual
                            self.pot += actual
                            any_action = True
                        print(f"  ✅ {player.name}: КОЛЛ ${actual} "
                              f"(лимит рейзов)", flush=True)
                    else:
                        print(f"  ✋ {player.name}: ЧЕК (лимит рейзов)",
                              flush=True)
                    time.sleep(PAUSE_ACTION)
                    continue

                game_state = {
                    'stage': stage,
                    'community_cards': self.community_cards,
                    'pot': self.pot,
                    'to_call': to_call,
                    'min_raise': min_raise,
                }

                action, amount, method, elapsed = player.make_decision(game_state)
                if not player.is_human():
                    style_info = f" [{player.style}]"

                if not human_folded and not player.is_human():
                    cards_info = f" [{player.cards[0]} {player.cards[1]}]"
                else:
                    cards_info = ""

                if action == "ФОЛД":
                    player.folded = True
                    print(f"  ❌ {player.name}{style_info}{cards_info}: ФОЛД  "
                          f"{method} ({elapsed:.2f}s)", flush=True)
                    any_action = True

                elif action == "ЧЕК":
                    print(f"  ✋ {player.name}{style_info}{cards_info}: ЧЕК  "
                          f"{method} ({elapsed:.2f}s)", flush=True)

                elif action == "КОЛЛ":
                    actual = min(to_call, player.stack)
                    if actual > 0:
                        player.stack -= actual
                        player.current_bet += actual
                        self.pot += actual
                        any_action = True
                        print(f"  ✅ {player.name}{style_info}{cards_info}: КОЛЛ ${actual}  "
                              f"{method} ({elapsed:.2f}s)", flush=True)
                    else:
                        print(f"  ✋ {player.name}{style_info}{cards_info}: ЧЕК  "
                              f"{method} ({elapsed:.2f}s)", flush=True)

                elif action == "РЕЙЗ":
                    total = min(amount, player.stack)
                    if total <= to_call:
                        actual = min(to_call, player.stack)
                        if actual > 0:
                            player.stack -= actual
                            player.current_bet += actual
                            self.pot += actual
                            any_action = True
                        print(f"  ✅ {player.name}{style_info}{cards_info}: КОЛЛ ${actual}  "
                              f"{method} ({elapsed:.2f}s)", flush=True)
                    else:
                        player.stack -= total
                        player.current_bet += total
                        self.pot += total
                        self.current_bet = player.current_bet
                        total_raises += 1
                        any_action = True
                        print(f"  🔥 {player.name}{style_info}{cards_info}: РЕЙЗ ${total}  "
                              f"{method} ({elapsed:.2f}s)", flush=True)

                time.sleep(PAUSE_ACTION)

                active = [p for p in alive
                          if not p.folded and not p.eliminated]
                if len(active) <= 1:
                    return

            # === КОНЕЦ ИТЕРАЦИИ ===
            active = [p for p in alive if not p.folded and not p.eliminated]
            if not active:
                return

            all_in = [p for p in active if p.stack == 0]
            if len(all_in) == len(active):
                return

            all_matched = all(
                p.current_bet >= self.current_bet for p in active
            )

            # === ГЛАВНЫЙ ФИКС ===
            # Если все уравняли — выходим, НЕ дожидаясь следующей итерации.
            # Это ломает зацикливание «лимит рейзов» → чек → лимит рейзов.
            if all_matched:
                if not any_action:
                    return          # никто не действовал — конец
                if total_raises >= MAX_RAISES_PER_ROUND:
                    return          # рейзы кончились, все уравняли — конец
                # Есть смысл дать ещё круг только если был рейз
                # (any_action=True и all_matched — это как раз ответ на рейз)
                # Но если в этой итерации не было рейзов, а только коллы —
                # тоже выходим.
                # Единственный случай, когда продолжаем — был рейз,
                # и не все успели ответить. Но это уже покрыто all_matched.
                # Так что выходим.
                return

        # Сработал MAX_ITERATIONS — принудительный выход
        print(f"  ⚠️ Достигнут лимит итераций в {stage} — выход из раунда",
              flush=True)
    def _showdown(self, alive: List[BasePlayer]):
        """Возвращает список победителей (может быть несколько при равных руках)."""
        active = [p for p in alive if not p.folded and not p.eliminated]
        if not active:
            return [], ""

        best_score = None
        winners = []

        for p in active:
            all_cards = p.cards + self.community_cards
            score = HandEvaluator.evaluate(all_cards)
            if best_score is None or score > best_score:
                best_score = score
                winners = [p]
            elif score == best_score:
                winners.append(p)

        winning_hand = HandEvaluator.category_name(best_score) if best_score else ""
        return winners, winning_hand

# ============================================================
# 8. ГЛАВНЫЙ ЦИКЛ
# ============================================================

def main():
    try:
        print("=" * 60)
        print("  🎰 ПОКЕР: ЧЕЛОВЕК ПРОТИВ ИИ")
        print("=" * 60)
        print("\n  Сначала убедись что сервер запущен:")
        print("  python model_server.py")
        print("\n  Сколько ИИ-игроков будет за столом?")
        print("  Минимум 1 (дуэль), максимум 8.")

        while True:
            try:
                ai_count = int(input("\n  ➤ Количество ИИ (1-8): ").strip())
                if 1 <= ai_count <= 8:
                    break
                print("  ⚠️ Введи число от 1 до 8.")
            except ValueError:
                print("  ⚠️ Введи целое число.")

        players = []

        human = HumanPlayer(name="ТЫ", stack=START_STACK)
        players.append(human)

        styles = [
        ("тайтовый",
         "Ты — ТАЙТОВЫЙ игрок. ПРАВИЛА: ФОЛД если сила руки ниже 0.35. "
         "КОЛЛ только с силой 0.5+ и коллом меньше 10% стека. "
         "РЕЙЗ только с силой 0.7+ и размером 30-50% банка."),
        ("лузовый",
         "Ты — ЛУЗОВЫЙ игрок. ПРАВИЛА: КОЛЛ почти всегда если колл меньше "
         "15% стека. ФОЛД только если сила ниже 0.15 И колл больше 15% "
         "стека. РЕЙЗ только с силой 0.75+."),
        ("агрессивный",
         "Ты — АГРЕССИВНЫЙ игрок. ПРАВИЛА: РЕЙЗ в 30% случаев с силой 0.5+, "
         "размер 40-60% банка. КОЛЛ с силой 0.35+. ФОЛД только с мусором "
         "ниже 0.2."),
        ("осторожный",
         "Ты — ОСТОРОЖНЫЙ игрок. ПРАВИЛА: ФОЛД если сила ниже 0.3 и есть "
         "ставка. КОЛЛ только с силой 0.4+ и коллом меньше 10% стека. "
         "РЕЙЗ только с силой 0.75+."),
        ("хитрый",
         "Ты — ХИТРЫЙ игрок. ПРАВИЛА: БЛЕФ в 15% случаев — РЕЙЗ 30% банка "
         "даже со слабой рукой 0.25. С сильной 0.7+ тоже РЕЙЗ. "
         "Со средней 0.3-0.6 — КОЛЛ."),
        ("математик",
         "Ты — МАТЕМАТИК. ПРАВИЛА: сила 0.6+ → РЕЙЗ 40% банка. "
         "0.4-0.6 → КОЛЛ если меньше 15% стека. Ниже 0.3 → ФОЛД."),
        ("безумный",
         "Ты — БЕЗУМНЫЙ игрок. ПРАВИЛА: РЕЙЗ в 50% случаев с силой 0.3+, "
         "размер 30-50% банка. ФОЛД только с мусором ниже 0.1."),
        ("профи",
         "Ты — ПРОФЕССИОНАЛ. ПРАВИЛА: 0.8+ → КОЛЛ или маленький РЕЙЗ 25% "
         "банка. 0.6-0.8 → РЕЙЗ 40%. 0.35-0.6 → ЧЕК или КОЛЛ. "
         "Ниже 0.3 → ФОЛД."),
    ]

        for i in range(ai_count):
            style, persona = styles[i % len(styles)]
            ai = AIPlayer(
                name=f"ИИ_{i+1}",
                stack=START_STACK,
                style=style,
                persona=persona,
            )
            players.append(ai)

        total_players = len(players)
        game = PokerGame(players)

        print("\n" + "=" * 60)
        print(f"  🎰 ПОКЕРНЫЙ СТОЛ: {total_players} ИГРОКОВ")
        print(f"  👤 Ты + {ai_count} ИИ-игроков")
        print(f"  Блайнды: ${SMALL_BLIND} / ${BIG_BLIND}")
        print(f"  Стек: ${START_STACK}")
        print("=" * 60)

        total_game_start = time.time()
        hand_num = 0
        human_elimination_shown = False

        while game.alive_count() > 1:
            hand_num += 1

            result = game.play_hand(hand_num)
            if result is None:
                break

            game.check_eliminations()

            # FIX: печатаем только один раз
            if human.eliminated and not human_elimination_shown:
                print("\n" + "=" * 60)
                print("  💔 Ты выбыл! Наблюдаем за ИИ...")
                print("=" * 60)
                human_elimination_shown = True

                alive_count = game.alive_count()
                if alive_count > 1:
                    game.dealer_pos = (game.dealer_pos + 1) % alive_count
                    time.sleep(PAUSE_STAGE)

        total_game_time = time.time() - total_game_start

        winner = next((p for p in players if not p.eliminated), None)

        print(f"\n{'╔' + '═'*76 + '╗'}", flush=True)
        if winner and winner.is_human():
            print(f"  🎉  ТЫ ВЫИГРАЛ ТУРНИР!", flush=True)
        elif winner:
            print(f"  🤖  ПОБЕДИТЕЛЬ: {winner.name}", flush=True)
        else:
            print(f"  🏁  ТУРНИР ЗАВЕРШЁН", flush=True)
        print(f"{'╚' + '═'*76 + '╝'}", flush=True)

        print(f"\n  {'Место':<6} {'Игрок':<15} {'Стек':<10} "
              f"{'Прибыль':<10} {'Статус':<12}", flush=True)
        print(f"  {'─'*60}", flush=True)

        place = 1
        sorted_players = sorted(players, key=lambda p: p.stack, reverse=True)
        for p in sorted_players:
            profit = p.stack - START_STACK
            sign = "+" if profit >= 0 else ""
            profit_str = f"{sign}{profit}"

            if p.eliminated:
                status = "💀 Выбыл"
            elif p.is_human():
                status = "🏆 Победитель"
            else:
                status = "🤖 Победитель"

            marker = " 👈" if p.is_human() else ""
            print(f"  #{place:<5} {p.name:<15} ${p.stack:<9} "
                  f"{profit_str:<10} {status}{marker}", flush=True)
            place += 1

        total_llm = sum(p.llm_calls for p in players if not p.is_human())

        print(f"\n  {'─'*60}", flush=True)
        print(f"  ⏱  Общее время: {total_game_time:.1f}s "
              f"({total_game_time/60:.1f} мин)", flush=True)
        print(f"  🎰 Раздач сыграно: {hand_num}", flush=True)
        print(f"  🧠 LLM-запросов: {total_llm}", flush=True)
        print(f"{'═'*78}", flush=True)
    except KeyboardInterrupt:
        print("\n\n  🛑 Игра прервана пользователем.", flush=True)
        # Показываем текущие стеки
        print(f"\n  💼 Финальные стеки:")
        print(f"  {'─'*55}", flush=True)
        for p in sorted(players, key=lambda x: x.stack, reverse=True):
            marker = " 👈" if p.is_human() else ""
            status = "💀" if p.eliminated else "🟢"
            print(f"  {status} {p.name:<12} ${p.stack}{marker}", flush=True)
        return

if __name__ == "__main__":
    main()