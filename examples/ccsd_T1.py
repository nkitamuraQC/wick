from wick.index import Idx
from wick.expression import *
from wick.hamiltonian import one_e, two_e, get_sym, commute
from wick.wick import apply_wick

H1 = one_e("f",["occ","vir"], norder=True)
H2 = two_e("I",["occ","vir"], norder=True)
H = H1 + H2

i = Idx(0,"occ")
a = Idx(0,"vir")
j = Idx(1,"occ")
b = Idx(1,"vir")
operators = [FOperator(i,True), FOperator(a,False)]
bra = Expression([Term(1.0, [], [Tensor([i,a],"")], operators, [])])
T1 = Expression([Term(1.0,
    [Sigma(i), Sigma(a)],
    [Tensor([a, i], "t")],
    [FOperator(a, True), FOperator(i, False)],
    [])])
sym = get_sym(True)
T2 = Expression([Term(0.25,
    [Sigma(i), Sigma(a), Sigma(j), Sigma(b)],
    [Tensor([a, b, i, j], "t",sym=sym)],
    [FOperator(a, True), FOperator(i, False), FOperator(b, True), FOperator(j, False)],
    [])])
T = T1 + T2

HT = commute(H,T)
HTT = commute(HT,T)
HTTT = commute(commute(commute(H2,T1),T1),T1)

S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.simplify()
final.sort()
print(final._print_str())
