# equity.py — точное эквити перебором + пот-оддсы
import itertools
from treys import Card, Evaluator

EV = Evaluator()
RANKS = "23456789TJQKA"; SUIT = {"♠":"s","♥":"h","♦":"d","♣":"c"}
FULL = [Card.new(r+s) for r in RANKS for s in "shdc"]

def c(t):                                   # "9♣" / "10♠" -> карта treys
    r = t[:-1]; r = "T" if r == "10" else r
    return Card.new(r + SUIT[t[-1]])

def equity(hero, board, villain):
    h, v, b = [c(x) for x in hero], [c(x) for x in villain], [c(x) for x in board]
    known = set(h + v + b)
    rest = [x for x in FULL if x not in known]
    need = 5 - len(b)
    win = tie = n = 0
    for run in itertools.combinations(rest, need):
        fb = b + list(run)
        s1, s2 = EV.evaluate(fb, h), EV.evaluate(fb, v)
        n += 1; win += s1 < s2; tie += s1 == s2
    return (win + tie/2) / n, n

def pot_odds(call, pot_after_bet):          # банк УЖЕ включает ставку оппонента
    return call / (pot_after_bet + call)

if __name__ == "__main__":
    hero, flop = ["9♣","8♣"], ["K♣","7♣","2♠"]
    eq, n = equity(hero, flop, ["K♦","Q♥"])
    print(f"флоп, 9♣8♣ vs K♦Q♥: эквити {eq*100:.2f}%  ({n} прогонов, точно)")
    turn = flop + ["2♦"]
    eq2, n2 = equity(hero, turn, ["K♦","Q♥"])
    print(f"тёрн +2♦:            эквити {eq2*100:.2f}%  ({n2} прогонов)")
    print(f"ставка 3bb в банк 4.5bb → пот-оддсы {pot_odds(3, 7.5)*100:.2f}%")
    for f in (0.25, 1/3, 0.5, 2/3, 1.0, 1.5):
        print(f"  ставка {f:.2f} банка → {pot_odds(f, 1+f)*100:.1f}%")