# poker_math.py   (pip install treys — чистый Python, ставится на Windows без проблем)
import random
from treys import Card, Evaluator

EV = Evaluator()
FULL = [Card.new(r + s) for r in "23456789TJQKA" for s in "shdc"]
SUIT = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}

def c(t):                      # "9♣" / "10♠" -> карта treys (твоя нотация из примера Г)
    return Card.new(t.replace("10", "T")[:-1] + SUIT[t[-1]])

def equity(hero, board, villain=None, n=10000):
    h, b = [c(x) for x in hero], [c(x) for x in board]
    v = [c(x) for x in villain] if villain else None
    known = set(h + b + (v or []))
    rest = [x for x in FULL if x not in known]
    win = tie = 0
    for _ in range(n):
        d = random.sample(rest, 2 if v else 4)
        full_board, opp = b + d[:2], (v if v else d[2:])
        s1, s2 = EV.evaluate(full_board, h), EV.evaluate(full_board, opp)
        win += s1 < s2         # в treys меньше очков = сильнее рука
        tie += s1 == s2
    return (win + tie / 2) / n

def pot_odds(call, pot_after_bet):
    return call / (pot_after_bet + call)

if __name__ == "__main__":
    hero, flop = ["9♣", "8♣"], ["K♣", "7♣", "2♠"]
    cart1, cart2 = "9♦", "7♥"
    print(f"эквити vs {cart1} {cart2}:       %.1f%%" % (100 * equity(hero, flop, [cart1, cart2])))
    print("эквити vs случайной:   %.1f%%" % (100 * equity(hero, flop)))
    print("пот-оддсы:             %.1f%%" % (100 * pot_odds(3, 4.5 + 3)))