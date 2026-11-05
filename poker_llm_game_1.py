"""
Покер с полным ИИ: 6 игроков на базе Gemma-3
10 раздач, блайнды 5/10, стек 1000
"""
import os
import re
import random
from itertools import combinations
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================
# 1. КАРТЫ И КОЛОДА
# ============================================================

@dataclass
class Card:
    rank: int   # 2-14 (11=J, 12=Q, 13=K, 14=A)
    suit: str   # ♠ ♥ ♦ ♣

    def __str__(self):
        rank_names = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        rank_str = rank_names.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit}"


class Deck:
    """Колода гарантирует уникальность карт."""
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
        """Раздаёт n карт, удаляя их из колоды."""
        return [self.cards.pop() for _ in range(n)]


# ============================================================
# 2. ОЦЕНЩИК РУК
# ============================================================

class HandEvaluator:
    CATEGORY_NAMES = {
        8: "Роял-флеш",
        7: "Стрит-флеш",
        6: "Каре",
        5: "Фулл-хаус",
        4: "Флеш",
        3: "Стрит",
        2: "Сет",
        1: "Две пары",
        0: "Пара",
        -1: "Старшая карта",
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
        """
        Оценка силы руки префлоп от 0.0 до 1.0.
        """
        if len(cards) < 2:
            return 0.0

        r1, r2 = sorted([cards[0].rank, cards[1].rank], reverse=True)
        suited = cards[0].suit == cards[1].suit

        score = 0.0

        # Пара
        if r1 == r2:
            score = 0.5 + (r1 - 2) / 24.0  # 2-2 = 0.5, A-A = 1.0
            return min(score, 1.0)

        # Высокие карты
        high_bonus = (r1 + r2) / 28.0 * 0.4
        score += high_bonus

        # Одномастные
        if suited:
            score += 0.1

        # Связанные (коннекторы)
        gap = r1 - r2
        if gap == 1:
            score += 0.1
        elif gap == 2:
            score += 0.05

        # Бонус за туза
        if r1 == 14:
            score += 0.1

        return min(max(score, 0.0), 1.0)

    @staticmethod
    def strength_description(strength: float) -> str:
        if strength >= 0.8:
            return "очень сильная"
        elif strength >= 0.6:
            return "сильная"
        elif strength >= 0.4:
            return "средняя"
        elif strength >= 0.2:
            return "слабая"
        else:
            return "очень слабая"


# ============================================================
# 3. ИГРОК НА БАЗЕ LLM
# ============================================================

class LLMPlayer:
    def __init__(self, name: str, stack: int, persona: str, model, tokenizer):
        self.name = name
        self.stack = stack
        self.persona = persona
        self.model = model
        self.tokenizer = tokenizer
        self.cards: List[Card] = []
        self.current_bet = 0
        self.folded = False
        self.bluff_instruction = ""

    def set_bluff_instruction(self, instruction: str):
        self.bluff_instruction = instruction

    def make_decision(self, game_state: dict) -> Tuple[str, int]:
        prompt = self._build_prompt(game_state)
        response = self._generate(prompt)
        action, amount = self._parse_decision(response, game_state)
        return action, amount

    def _build_prompt(self, game_state: dict) -> str:
        cards_str = " ".join(str(c) for c in self.cards)
        community = game_state['community_cards']
        community_str = " ".join(str(c) for c in community) if community else "нет"
        stage = game_state['stage']

        # Оценка силы руки
        if community:
            all_cards = self.cards + community
            score = HandEvaluator.evaluate(all_cards)
            hand_name = HandEvaluator.category_name(score)
            strength = (score[0] + 1) / 9.0  # Нормализация -1..8 -> 0..1
        else:
            strength = HandEvaluator.preflop_strength(self.cards)
            hand_name = "префлоп"

        strength_desc = HandEvaluator.strength_description(strength)

        prompt = f"""{self.persona}

"""
        if self.bluff_instruction:
            prompt += f"""⚠️ ОБЯЗАТЕЛЬНАЯ ИНСТРУКЦИЯ:
{self.bluff_instruction}

"""

        prompt += f"""=== СИТУАЦИЯ ===
Стадия: {stage}
Твои карты: {cards_str}
Карты на столе: {community_str}
Твоя комбинация: {hand_name}
Сила руки: {strength:.2f} ({strength_desc})
Банк: ${game_state['pot']}
Нужно уравнять: ${game_state['to_call']}
Твой стек: ${self.stack}

=== ДЕЙСТВИЯ ===
- ФОЛД: сбросить карты
- КОЛЛ {game_state['to_call']}: уравнять
- РЕЙЗ {game_state['min_raise']}: повысить (минимум ${game_state['min_raise']})

=== ОТВЕТ ===
Напиши ровно одну строку в формате:
ДЕЙСТВИЕ: [ФОЛД/КОЛЛ/РЕЙЗ] | СУММА: [число] | ПРИЧИНА: [почему]
"""
        return prompt

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def _parse_decision(self, response: str, game_state: dict) -> Tuple[str, int]:
        upper = response.upper()

        if "ФОЛД" in upper or "FOLD" in upper or "СБРОС" in upper:
            action = "ФОЛД"
        elif "РЕЙЗ" in upper or "RAISE" in upper or "ПОВЫШ" in upper:
            action = "РЕЙЗ"
        elif "КОЛЛ" in upper or "CALL" in upper or "УРАВН" in upper:
            action = "КОЛЛ"
        else:
            action = "КОЛЛ"  # По умолчанию колл, а не фолд

        numbers = re.findall(r'\d+', response)
        amount = int(numbers[-1]) if numbers else 0

        if action == "КОЛЛ":
            amount = min(game_state['to_call'], self.stack)
        elif action == "РЕЙЗ":
            amount = max(amount, game_state['min_raise'])
            amount = min(amount, self.stack)

        return action, amount


# ============================================================
# 4. ИГРОВОЙ МЕНЕДЖЕР
# ============================================================

class PokerGame:
    def __init__(self, players: List[LLMPlayer], small_blind=5, big_blind=10):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = Deck()
        self.community_cards: List[Card] = []
        self.pot = 0
        self.current_bet = 0
        self.dealer_pos = 0

    def play_hand(self, hand_number: int) -> dict:
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0

        for player in self.players:
            player.cards = self.deck.deal(2)
            player.current_bet = 0
            player.folded = False
            if hand_number < 10:
                player.bluff_instruction = ""

        # Блайнды
        sb_pos = (self.dealer_pos + 1) % len(self.players)
        bb_pos = (self.dealer_pos + 2) % len(self.players)

        self._post_blind(self.players[sb_pos], self.small_blind)
        self._post_blind(self.players[bb_pos], self.big_blind)
        self.current_bet = self.big_blind

        result = {
            'hand_number': hand_number,
            'players': [(p.name, str(p.cards[0]), str(p.cards[1])) for p in self.players],
            'community': [],
            'actions': [],
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
                result['community'] = [str(c) for c in self.community_cards]

            self._betting_round(stage_name, result)

            active = [p for p in self.players if not p.folded]
            if len(active) == 1:
                winner = active[0]
                winner.stack += self.pot
                result['winner'] = winner.name
                result['pot'] = self.pot
                result['winning_hand'] = "Все соперники сбросили"
                return result

            if len(active) == 0:
                # Все сбросили — банк уходит большому блайнду
                bb_player = self.players[bb_pos]
                bb_player.stack += self.pot
                result['winner'] = bb_player.name
                result['pot'] = self.pot
                result['winning_hand'] = "Все сбросили (блайнд забирает)"
                return result

        winner, winning_hand = self._showdown()
        if winner:
            winner.stack += self.pot
            result['winner'] = winner.name
            result['pot'] = self.pot
            result['winning_hand'] = winning_hand
        else:
            # Никто не выиграл — делим банк
            active = [p for p in self.players if not p.folded]
            if active:
                share = self.pot // len(active)
                for p in active:
                    p.stack += share
                result['winner'] = "Ничья"
                result['pot'] = self.pot

        return result

    def _post_blind(self, player: LLMPlayer, amount: int):
        actual = min(amount, player.stack)
        player.stack -= actual
        player.current_bet = actual
        self.pot += actual

    def _betting_round(self, stage: str, result: dict):
        active = [p for p in self.players if not p.folded]
        if not active:
            return

        for player in active:
            to_call = self.current_bet - player.current_bet
            min_raise = self.current_bet + self.big_blind

            game_state = {
                'stage': stage,
                'community_cards': self.community_cards,
                'pot': self.pot,
                'to_call': to_call,
                'min_raise': min_raise,
            }

            action, amount = player.make_decision(game_state)

            if action == "ФОЛД":
                player.folded = True
                result['actions'].append((player.name, stage, "ФОЛД", 0))

            elif action == "КОЛЛ":
                actual = min(to_call, player.stack)
                if actual > 0:
                    player.stack -= actual
                    player.current_bet += actual
                    self.pot += actual
                result['actions'].append((player.name, stage, "КОЛЛ", actual))

            elif action == "РЕЙЗ":
                total = min(amount, player.stack)
                if total > 0:
                    player.stack -= total
                    player.current_bet += total
                    self.pot += total
                    self.current_bet = player.current_bet
                result['actions'].append((player.name, stage, "РЕЙЗ", total))

    def _showdown(self) -> Tuple[Optional[LLMPlayer], str]:
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

def print_hand_result(result: dict, players: List[LLMPlayer]):
    hand_num = result['hand_number']

    print(f"\n{'═'*70}")
    print(f"  РАЗДАЧА #{hand_num}")
    print(f"{'═'*70}")

    # Игроки и их карты
    print(f"\n  {'Игрок':<15} {'Карты':<12} {'Стек':<10}")
    print(f"  {'─'*50}")
    for name, c1, c2 in result['players']:
        stack = next(p.stack for p in players if p.name == name)
        print(f"  {name:<15} {c1} {c2:<8} ${stack}")

    # Карты на столе
    if result['community']:
        community_str = " ".join(result['community'])
        print(f"\n  🃏 Карты на столе: {community_str}")

    # Действия
    print(f"\n  Ход раздачи:")
    print(f"  {'─'*50}")
    for name, stage, action, amount in result['actions']:
        if amount > 0:
            print(f"  [{stage:>7}] {name}: {action} ${amount}")
        else:
            print(f"  [{stage:>7}] {name}: {action}")

    # Победитель
    print(f"\n  🏆 Победитель: {result['winner']}")
    print(f"  💰 Банк: ${result['pot']}")
    if result['winning_hand']:
        print(f"  🃏 Комбинация: {result['winning_hand']}")
    print(f"{'═'*70}")


# ============================================================
# 6. ЗАПУСК
# ============================================================

def main():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN не найден. Проверь .env")

    MODEL_ID = "google/gemma-3-4b-it"

    print("=" * 70)
    print("  ЗАГРУЗКА МОДЕЛИ")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        token=HF_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("✅ Модель загружена\n")

    # Персонажи (более агрессивные)
    personas = [
        (
            "Ты — тайтово-агрессивный игрок. Играешь с парами, "
            "двумя высокими картами (A, K, Q), одномастными коннекторами. "
            "С мусором (разномастные низкие карты) — ФОЛД. "
            "С хорошей рукой — РЕЙЗ. Ты НЕ боишься ставить."
        ),
        (
            "Ты — лузовый игрок. Коллируешь почти всегда, даже со слабыми картами. "
            "ФОЛД делаешь только если ставка больше 30% твоего стека. "
            "Любишь смотреть флоп. Часто делаешь КОЛЛ."
        ),
        (
            "Ты — агрессивный игрок. Делаешь РЕЙЗ в 60% случаев. "
            "Даже со средней рукой предпочитаешь РЕЙЗ вместо КОЛЛ. "
            "ФОЛД делаешь крайне редко. Твоя стратегия — давление."
        ),
        (
            "Ты — осторожный игрок, но не трусливый. "
            "С парой или двумя высокими картами — КОЛЛ. "
            "С сильной рукой (две пары, сет) — РЕЙЗ. "
            "ФОЛД только с совсем мусорными картами."
        ),
        (
            "Ты — хитрый игрок. В 30% случаев делаешь РЕЙЗ даже со слабой рукой (блеф). "
            "С сильной рукой тоже РЕЙЗ. Любишь запутывать соперников. "
            "ФОЛД делаешь редко."
        ),
        (
            "Ты — математик. Если сила руки выше 0.4 — КОЛЛ или РЕЙЗ. "
            "Если сила руки выше 0.7 — обязательно РЕЙЗ. "
            "ФОЛД только если сила руки ниже 0.2 и ставка большая. "
            "Ты играешь рационально, но НЕ фолдишь всё подряд."
        ),
    ]

    players = []
    for i, persona in enumerate(personas):
        player = LLMPlayer(
            name=f"Игрок_{i+1}",
            stack=1000,
            persona=persona,
            model=model,
            tokenizer=tokenizer,
        )
        players.append(player)

    game = PokerGame(players, small_blind=5, big_blind=10)

    print("=" * 70)
    print("  ПОКЕРНЫЙ СТОЛ: 6 ИГРОКОВ-ИИ")
    print("  Блайнды: $5 / $10 | Стек: $1000 | Раздач: 10")
    print("=" * 70)

    for hand_num in range(1, 11):
        if hand_num == 10:
            bluffer = random.choice(players)
            bluffer.set_bluff_instruction(
                "На этой раздаче ты ОБЯЗАН блефовать! "
                "Сделай РЕЙЗ минимум $100, даже если у тебя слабая рука. "
                "Цель — заставить соперников сбросить. НЕ ФОЛДИ!"
            )
            print(f"\n⚠️  РАЗДАЧА #10: {bluffer.name} получает инструкцию БЛЕФОВАТЬ!")

        result = game.play_hand(hand_num)
        print_hand_result(result, players)

        game.dealer_pos = (game.dealer_pos + 1) % len(players)

    # Финальные стеки
    print(f"\n{'═'*70}")
    print("  ИТОГОВЫЕ СТЕКИ")
    print(f"{'═'*70}")
    for player in sorted(players, key=lambda p: p.stack, reverse=True):
        emoji = "🏆" if player.stack == max(p.stack for p in players) else "  "
        print(f"  {emoji} {player.name:<15} ${player.stack}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()