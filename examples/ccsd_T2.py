from wick.expression import *
from wick.ops import *
from wick.wick import apply_wick

H1 = one_e("f",["occ","vir"], norder=True)
H2 = two_e("I",["occ","vir"], norder=True, compress=True)
H = H1 + H2

bra = braE2("occ", "vir", "occ", "vir")
T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])
T = T1 + T2

HT = commute(H,T)
HTT = commute(HT,T)
HTTT = commute(HTT,T)
HTTTT = commute(HTTT,T)

S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
