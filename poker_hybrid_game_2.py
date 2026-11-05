"""
Покер с гибридным ИИ: правила + Gemma-3
- Правильный порядок ходов по кругу
- Битва ставок (ре-рейзы до уравнения)
- Оценка дро-потенциала (флеш-дро, стрит-дро)
- Таймеры для отслеживания скорости
"""
import os
import re
import random
import time
from itertools import combinations
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# 1. КАРТЫ И КОЛОДА
# ============================================================

@dataclass
class Card:
    rank: int
    suit: str

    def __str__(self):
        rank_names = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        rank_str = rank_names.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit}"


class Deck:
    def __init__(self):
        self.cards = []
        self.reset()

    def reset(self):
        self.cards = [
            Card(rank, suit)
            for rank in range(2, 15)
            for suit in ['♠', '♥', '♦', '♣']
        ]
        random.shuffle(self.cards)

    def deal(self, n=1):
        return [self.cards.pop() for _ in range(n)]


# ============================================================
# 2. ОЦЕНЩИК РУК + ДРО-ПОТЕНЦИАЛ
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
            score = HandEvaluator._evaluate_5(list(combo))
            if best is None or score > best:
                best = score
        return best

    @staticmethod
    def _evaluate_5(cards: List[Card]) -> Tuple[int, tuple]:
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        is_flush = len(set(suits)) == 1

        is_straight = False
        straight_high = 0
        unique = sorted(set(ranks), reverse=True)
        if len(unique) == 5:
            if unique[0] - unique[4] == 4:
                is_straight = True
                straight_high = unique[0]
            elif unique == [14, 5, 4, 3, 2]:
                is_straight = True
                straight_high = 5

        counts = Counter(ranks)
        count_values = sorted(counts.values(), reverse=True)

        if is_flush and is_straight:
            category = 8 if straight_high == 14 else 7
        elif count_values[0] == 4:
            category = 6
        elif count_values[0] == 3 and count_values[1] == 2:
            category = 5
        elif is_flush:
            category = 4
        elif is_straight:
            category = 3
        elif count_values[0] == 3:
            category = 2
        elif count_values[0] == 2 and count_values[1] == 2:
            category = 1
        elif count_values[0] == 2:
            category = 0
        else:
            category = -1

        kickers = []
        for count in sorted(set(count_values), reverse=True):
            for rank, cnt in counts.items():
                if cnt == count:
                    kickers.append(rank)

        return (category, tuple(sorted(kickers, reverse=True)))

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
            score = 0.5 + (r1 - 2) / 24.0
            return min(score, 1.0)
        score += (r1 + r2) / 28.0 * 0.4
        if suited:
            score += 0.1
        gap = r1 - r2
        if gap == 1:
            score += 0.1
        elif gap == 2:
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
        """
        Оценивает потенциал улучшения руки (дро).
        Возвращает (бонус к силе, описание дро).
        """
        all_cards = cards + community
        if len(all_cards) < 5:
            return 0.0, ""

        bonus = 0.0
        draws = []

        # Флеш-дро: 4 карты одной масти
        suit_counts = Counter(c.suit for c in all_cards)
        for suit, count in suit_counts.items():
            if count == 4:
                bonus += 0.20
                draws.append("флеш-дро")
                break

        # Стрит-дро: 4 последовательные карты из 5 возможных
        ranks = sorted(set(c.rank for c in all_cards))
        # Проверяем каждое окно из 5 последовательных рангов
        for start in range(2, 11):  # 2..10
            window = set(range(start, start + 5))
            # Учитываем туз как единицу для A-2-3-4-5
            if start == 10:
                window = {10, 11, 12, 13, 14}

            overlap = len(window.intersection(ranks))
            if overlap == 4:
                bonus += 0.15
                draws.append("стрит-дро")
                break

        # Гатшот: 4 из 5 с дыркой посередине
        if not draws:
            for start in range(2, 11):
                window = set(range(start, start + 5))
                overlap = len(window.intersection(ranks))
                if overlap == 4:
                    # Проверяем что дырка не на краю (это было бы стрит-дро)
                    missing = window - ranks
                    if missing:
                        missing_rank = list(missing)[0]
                        if missing_rank != start and missing_rank != start + 4:
                            bonus += 0.10
                            draws.append("гатшот")
                            break

        # Туз в руке + флеш-дро = сильнее
        if "флеш-дро" in draws and any(c.rank == 14 for c in cards):
            bonus += 0.05

        draw_desc = ", ".join(draws) if draws else ""
        return min(bonus, 0.35), draw_desc

    @staticmethod
    def effective_strength(cards: List[Card], community: List[Card],
                           stage: str) -> Tuple[float, str]:
        """
        Эффективная сила руки = текущая сила + дро-потенциал.
        Возвращает (сила, описание).
        """
        if not community:
            strength = HandEvaluator.preflop_strength(cards)
            return strength, "префлоп"

        base_strength = HandEvaluator.postflop_strength(cards, community)
        draw_bonus, draw_desc = HandEvaluator.draw_potential(cards, community)

        # На ривере дро уже не имеет смысла
        if stage == "Ривер":
            draw_bonus = 0.0
            draw_desc = ""

        effective = min(base_strength + draw_bonus, 1.0)

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
# 3. ГИБРИДНЫЙ ИГРОК
# ============================================================

class HybridPlayer:
    def __init__(self, name, stack, style, model=None, tokenizer=None,
                 big_blind=10):
        self.name = name
        self.stack = stack
        self.style = style
        self.model = model
        self.tokenizer = tokenizer
        self.big_blind = big_blind
        self.cards: List[Card] = []
        self.current_bet = 0
        self.folded = False
        self.bluff_instruction = ""
        self.llm_calls = 0
        self.rule_calls = 0

    def set_bluff_instruction(self, instruction: str):
        self.bluff_instruction = instruction

    def reset_hand(self):
        self.cards = []
        self.current_bet = 0
        self.folded = False

    def make_decision(self, game_state: dict) -> Tuple[str, int, str, float]:
        start = time.time()

        # Принудительный блеф — всегда LLM
        if self.bluff_instruction:
            action, amount = self._llm_decision(game_state)
            elapsed = time.time() - start
            self.llm_calls += 1
            return action, amount, "🧠 LLM (блеф)", elapsed

        # Быстрое решение
        action, amount = self._rule_decision(game_state)
        if action is not None:
            elapsed = time.time() - start
            self.rule_calls += 1
            return action, amount, "⚡ Правило", elapsed

        # Сложный случай — LLM
        action, amount = self._llm_decision(game_state)
        elapsed = time.time() - start
        self.llm_calls += 1
        return action, amount, "🧠 LLM", elapsed

    def _rule_decision(self, game_state: dict) -> Tuple[Optional[str], int]:
        community = game_state['community_cards']
        to_call = game_state['to_call']
        pot = game_state['pot']
        stage = game_state['stage']

        # Эффективная сила руки (с учётом дро)
        strength, desc = HandEvaluator.effective_strength(
            self.cards, community, stage
        )

        # === ЕСЛИ НЕ НУЖНО ПЛАТИТЬ ===
        if to_call == 0:
            if strength >= 0.65:
                bet = min(int(pot * 0.6) + self.big_blind, self.stack)
                return "РЕЙЗ", max(bet, self.big_blind)
            elif strength >= 0.35:
                # Иногда ставим с дро
                if "дро" in desc and random.random() < 0.4:
                    bet = min(int(pot * 0.4) + self.big_blind, self.stack)
                    return "РЕЙЗ", max(bet, self.big_blind)
                return "ЧЕК", 0
            else:
                return "ЧЕК", 0

        # === ОЧЕНЬ СИЛЬНАЯ РУКА → РЕЙЗ ===
        if strength >= 0.75:
            raise_amount = min(int(pot * 0.75) + self.big_blind, self.stack)
            raise_amount = max(raise_amount, game_state['min_raise'])
            return "РЕЙЗ", raise_amount

        # === ХОРОШАЯ РУКА → КОЛЛ или РЕЙЗ ===
        if strength >= 0.55:
            if to_call <= self.stack * 0.25:
                # Иногда рейзим с хорошей рукой
                if self.style == "агрессивный" and random.random() < 0.3:
                    raise_amount = min(int(pot * 0.5) + self.big_blind, self.stack)
                    return "РЕЙЗ", max(raise_amount, game_state['min_raise'])
                return "КОЛЛ", to_call
            else:
                return "ФОЛД", 0

        # === СРЕДНЯЯ РУКА / ДРО ===
        if strength >= 0.35:
            if to_call <= self.stack * 0.1:
                return "КОЛЛ", to_call
            elif to_call <= self.big_blind * 2:
                # Дешёвый колл с дро
                if "дро" in desc:
                    return "КОЛЛ", to_call
                return "ФОЛД", 0
            else:
                return "ФОЛД", 0

        # === СЛАБАЯ РУКА ===
        if strength < 0.2:
            if to_call > self.stack * 0.1:
                return "ФОЛД", 0
            elif to_call <= self.big_blind:
                if self.style in ["лузовый", "агрессивный", "хитрый"]:
                    return "КОЛЛ", to_call
                return "ФОЛД", 0
            else:
                return "ФОЛД", 0

        # === ОСТАЛЬНОЕ → LLM ===
        return None, 0

    def _llm_decision(self, game_state: dict) -> Tuple[str, int]:
        if self.model is None or self.tokenizer is None:
            return self._fallback_decision(game_state)

        prompt = self._build_prompt(game_state)
        response = self._generate(prompt)
        return self._parse_decision(response, game_state)

    def _fallback_decision(self, game_state: dict) -> Tuple[str, int]:
        community = game_state['community_cards']
        stage = game_state.get('stage', '')
        strength, _ = HandEvaluator.effective_strength(
            self.cards, community, stage
        )
        if strength >= 0.5:
            return "КОЛЛ", game_state['to_call']
        elif strength >= 0.3:
            if random.random() < 0.5:
                return "КОЛЛ", game_state['to_call']
            return "ФОЛД", 0
        return "ФОЛД", 0

    def _build_prompt(self, game_state: dict) -> str:
        cards_str = " ".join(str(c) for c in self.cards)
        community = game_state['community_cards']
        community_str = " ".join(str(c) for c in community) if community else "нет"
        stage = game_state['stage']

        strength, desc = HandEvaluator.effective_strength(
            self.cards, community, stage
        )
        strength_word = HandEvaluator.strength_desc(strength)

        prompt = f"""Ты — покерный игрок со стилем: {self.style}.

"""
        if self.bluff_instruction:
            prompt += f"""⚠️ {self.bluff_instruction}

"""

        prompt += f"""Твои карты: {cards_str}
На столе: {community_str}
Рука: {desc}
Сила: {strength:.2f} ({strength_word})
Банк: ${game_state['pot']}
Колл: ${game_state['to_call']}
Стек: ${self.stack}

Ответь ОДНОЙ строкой:
ДЕЙСТВИЕ: [ФОЛД/КОЛЛ/РЕЙЗ] | СУММА: [число]
"""
        return prompt

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def _parse_decision(self, response: str, game_state: dict) -> Tuple[str, int]:
        upper = response.upper()

        if "ФОЛД" in upper or "FOLD" in upper:
            action = "ФОЛД"
        elif "РЕЙЗ" in upper or "RAISE" in upper or "СТАВК" in upper:
            action = "РЕЙЗ"
        elif "ЧЕК" in upper or "CHECK" in upper:
            action = "ЧЕК"
        elif "КОЛЛ" in upper or "CALL" in upper:
            action = "КОЛЛ"
        else:
            action = "КОЛЛ"

        numbers = re.findall(r'\d+', response)
        amount = int(numbers[-1]) if numbers else 0

        if action == "КОЛЛ":
            amount = min(game_state['to_call'], self.stack)
        elif action == "РЕЙЗ":
            amount = max(amount, game_state['min_raise'])
            amount = min(amount, self.stack)
        elif action == "ЧЕК":
            amount = 0

        return action, amount


# ============================================================
# 4. ИГРОВОЙ МЕНЕДЖЕР С БИТВОЙ СТАВОК
# ============================================================

class PokerGame:
    def __init__(self, players: List[HybridPlayer], small_blind=5, big_blind=10):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = Deck()
        self.community_cards: List[Card] = []
        self.pot = 0
        self.current_bet = 0
        self.dealer_pos = 0

    def _get_action_order(self) -> List[HybridPlayer]:
        """
        Порядок ходов по кругу, начиная после большого блайнда.
        Позиция дилера сдвигается каждую раздачу.
        """
        n = len(self.players)
        # Первый ходит игрок после большого блайнда
        start = (self.dealer_pos + 3) % n
        order = []
        for i in range(n):
            order.append(self.players[(start + i) % n])
        return order

    def play_hand(self, hand_number: int) -> dict:
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0

        for player in self.players:
            player.reset_hand()
            player.cards = self.deck.deal(2)
            if hand_number < 10:
                player.bluff_instruction = ""

        # Блайнды
        n = len(self.players)
        sb_pos = (self.dealer_pos + 1) % n
        bb_pos = (self.dealer_pos + 2) % n

        self._post_blind(self.players[sb_pos], self.small_blind)
        self._post_blind(self.players[bb_pos], self.big_blind)
        self.current_bet = self.big_blind

        result = {
            'hand_number': hand_number,
            'players': [(p.name, str(p.cards[0]), str(p.cards[1]))
                        for p in self.players],
            'community': [],
            'actions': [],
            'winner': None,
            'pot': 0,
            'winning_hand': "",
            'total_time': 0,
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
                result['community'] = [str(c) for c in self.community_cards]

            # Пропускаем раунд если остался 1 игрок
            active = [p for p in self.players if not p.folded]
            if len(active) > 1:
                self._betting_round(stage_name, result)

            # Проверяем после раунда
            active = [p for p in self.players if not p.folded]
            if len(active) == 1:
                winner = active[0]
                winner.stack += self.pot
                result['winner'] = winner.name
                result['pot'] = self.pot
                result['winning_hand'] = "Все сбросили"
                return result

            if len(active) == 0:
                bb_player = self.players[bb_pos]
                bb_player.stack += self.pot
                result['winner'] = bb_player.name
                result['pot'] = self.pot
                result['winning_hand'] = "Все сбросили (блайнд)"
                return result

        winner, winning_hand = self._showdown()
        if winner:
            winner.stack += self.pot
            result['winner'] = winner.name
            result['pot'] = self.pot
            result['winning_hand'] = winning_hand

        return result

    def _post_blind(self, player: HybridPlayer, amount: int):
        actual = min(amount, player.stack)
        player.stack -= actual
        player.current_bet = actual
        self.pot += actual

    def _betting_round(self, stage: str, result: dict):
        """
        Битва ставок: раунд продолжается до тех пор,
        пока все активные игроки не уравняют ставку или не сбросят.
        Максимум 10 итераций для защиты от бесконечного цикла.
        """
        action_order = self._get_action_order()
        max_iterations = 10

        for iteration in range(max_iterations):
            acted = False

            for player in action_order:
                if player.folded:
                    continue

                to_call = self.current_bet - player.current_bet
                min_raise = self.current_bet + self.big_blind

                game_state = {
                    'stage': stage,
                    'community_cards': self.community_cards,
                    'pot': self.pot,
                    'to_call': to_call,
                    'min_raise': min_raise,
                }

                action, amount, method, elapsed = player.make_decision(game_state)

                if action == "ФОЛД":
                    player.folded = True
                    result['actions'].append(
                        (player.name, stage, "ФОЛД", 0, method, elapsed)
                    )
                    acted = True

                elif action == "ЧЕК":
                    result['actions'].append(
                        (player.name, stage, "ЧЕК", 0, method, elapsed)
                    )

                elif action == "КОЛЛ":
                    actual = min(to_call, player.stack)
                    if actual > 0:
                        player.stack -= actual
                        player.current_bet += actual
                        self.pot += actual
                    result['actions'].append(
                        (player.name, stage, "КОЛЛ", actual, method, elapsed)
                    )
                    if actual > 0:
                        acted = True

                elif action == "РЕЙЗ":
                    total = min(amount, player.stack)
                    if total > 0:
                        player.stack -= total
                        player.current_bet += total
                        self.pot += total
                        self.current_bet = player.current_bet
                    result['actions'].append(
                        (player.name, stage, "РЕЙЗ", total, method, elapsed)
                    )
                    acted = True

                # Если остался один игрок — выходим
                active = [p for p in self.players if not p.folded]
                if len(active) <= 1:
                    return

            # Проверяем: все активные уравняли?
            active = [p for p in self.players if not p.folded]
            if not active:
                return

            all_matched = all(
                p.current_bet >= self.current_bet for p in active
            )

            if all_matched and not acted:
                break

            if all_matched:
                break

    def _showdown(self) -> Tuple[Optional[HybridPlayer], str]:
        active = [p for p in self.players if not p.folded]
        if not active:
            return None, ""

        best_score = None
        winner = None

        for player in active:
            all_cards = player.cards + self.community_cards
            score = HandEvaluator.evaluate(all_cards)
            if best_score is None or score > best_score:
                best_score = score
                winner = player

        winning_hand = HandEvaluator.category_name(best_score) if best_score else ""
        return winner, winning_hand


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def print_hand_result(result: dict, players: List[HybridPlayer]):
    hand_num = result['hand_number']

    print(f"\n{'═'*78}")
    print(f"  РАЗДАЧА #{hand_num}")
    print(f"{'═'*78}")

    print(f"\n  {'Игрок':<15} {'Карты':<12} {'Стек':<10}")
    print(f"  {'─'*50}")
    for name, c1, c2 in result['players']:
        stack = next(p.stack for p in players if p.name == name)
        print(f"  {name:<15} {c1} {c2:<8} ${stack}")

    if result['community']:
        community_str = " ".join(result['community'])
        print(f"\n  🃏 Стол: {community_str}")

    print(f"\n  Ход раздачи:")
    print(f"  {'─'*75}")
    print(f"  {'Игрок':<12} {'Стадия':<8} {'Действие':<8} {'$':<7} "
          f"{'Метод':<18} {'Время'}")
    print(f"  {'─'*75}")

    total_hand_time = 0
    current_stage = ""
    for name, stage, action, amount, method, elapsed in result['actions']:
        total_hand_time += elapsed
        time_str = f"{elapsed:.3f}s"

        # Разделитель между стадиями
        if stage != current_stage:
            if current_stage:
                print(f"  {'·'*75}")
            current_stage = stage

        if amount > 0:
            print(f"  {name:<12} {stage:<8} {action:<8} ${amount:<6} "
                  f"{method:<18} {time_str}")
        else:
            print(f"  {name:<12} {stage:<8} {action:<8} {'—':<7} "
                  f"{method:<18} {time_str}")

    result['total_time'] = total_hand_time

    print(f"\n  🏆 Победитель: {result['winner']}")
    print(f"  💰 Банк: ${result['pot']}")
    if result['winning_hand']:
        print(f"  🃏 Комбинация: {result['winning_hand']}")
    print(f"  ⏱  Время раздачи: {total_hand_time:.2f}s")
    print(f"{'═'*78}")


# ============================================================
# 6. ЗАПУСК
# ============================================================

def main():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN не найден")

    MODEL_ID = "google/gemma-3-4b-it"
    LORA_PATH = "./poker_gemma3_lora"

    print("=" * 78)
    print("  ЗАГРУЗКА МОДЕЛИ")
    print("=" * 78)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        token=HF_TOKEN,
    )

    if os.path.exists(LORA_PATH):
        print("Подключение обученного LoRA-адаптера...")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
        print(f"✅ LoRA подключен из {LORA_PATH}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        print("⚠️ LoRA не найден, используем базовую модель")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_time = time.time() - load_start
    print(f"✅ Модель загружена за {load_time:.1f}s\n")

    styles = [
        ("тайтовый", "Тайтовый. Играет только с хорошими руками."),
        ("лузовый", "Лузовый. Коллирует почти всегда."),
        ("агрессивный", "Агрессивный. Часто рейзит и давит."),
        ("осторожный", "Осторожный. Редко блефует."),
        ("хитрый", "Хитрый. Любит блефовать и ловить."),
        ("математик", "Математик. Играет по вероятностям."),
    ]

    players = []
    for i, (style, persona) in enumerate(styles):
        player = HybridPlayer(
            name=f"Игрок_{i+1}",
            stack=1000,
            style=style,
            model=model,
            tokenizer=tokenizer,
            big_blind=10,
        )
        players.append(player)

    game = PokerGame(players, small_blind=5, big_blind=10)

    print("=" * 78)
    print("  ПОКЕРНЫЙ СТОЛ: 6 ИГРОКОВ (ГИБРИДНЫЙ ИИ)")
    print("  Блайнды: $5 / $10 | Стек: $1000 | Раздач: 10")
    print("  ⚡ = правило | 🧠 = LLM | Дро = потенциал улучшения")
    print("=" * 78)

    total_game_start = time.time()

    for hand_num in range(1, 11):
        if hand_num == 10:
            bluffer = random.choice(players)
            bluffer.set_bluff_instruction(
                "ОБЯЗАТЕЛЬНО блефуй! Сделай РЕЙЗ $100+ даже со слабой рукой. НЕ ФОЛДИ!"
            )
            print(f"\n⚠️  РАЗДАЧА #10: {bluffer.name} получает инструкцию БЛЕФОВАТЬ!")

        result = game.play_hand(hand_num)
        print_hand_result(result, players)

        game.dealer_pos = (game.dealer_pos + 1) % len(players)

    total_game_time = time.time() - total_game_start

    # Итоги
    print(f"\n{'═'*78}")
    print("  ИТОГОВЫЕ СТЕКИ")
    print(f"{'═'*78}")
    for player in sorted(players, key=lambda p: p.stack, reverse=True):
        emoji = "🏆" if player.stack == max(p.stack for p in players) else "  "
        print(f"  {emoji} {player.name:<15} ${player.stack}")

    total_llm = sum(p.llm_calls for p in players)
    total_rule = sum(p.rule_calls for p in players)
    total_calls = total_llm + total_rule

    print(f"\n{'═'*78}")
    print("  СТАТИСТИКА")
    print(f"{'═'*78}")
    print(f"  Общее время: {total_game_time:.1f}s ({total_game_time/60:.1f} мин)")
    print(f"  Среднее время на раздачу: {total_game_time/10:.1f}s")
    print(f"  LLM-запросов: {total_llm}")
    print(f"  Решений по правилам: {total_rule}")
    if total_calls > 0:
        print(f"  Доля LLM: {total_llm/total_calls*100:.0f}%")
    print(f"{'═'*78}")


if __name__ == "__main__":
    main()