from fractions import Fraction
from wick.index import Idx
from wick.operator import FOperator, Tensor
from wick.expression import Term, Expression, AExpression
from wick.wick import apply_wick
from wick.convenience import E1, E2, commute, ketE1, ketE2

i = Idx(0, "occ")
a = Idx(0, "vir")
j = Idx(1, "occ")
b = Idx(1, "vir")

T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])
T = T1 + T2

L1 = E1("L", ["vir"], ["occ"])
L2 = E2("L", ["vir"], ["occ"])
L = L1 + L2

# ov block
operators = [FOperator(a, True), FOperator(i, False)]
pvo = Expression([Term(1, [], [Tensor([i, a], "")], operators, [])])
ket1 = ketE1("occ", "vir")
ket2 = ketE2("occ", "vir", "occ", "vir")

PT = commute(pvo, T)
PTT = commute(PT, T)
mid = pvo + PT + Fraction('1/2')*PTT
full1 = L*mid*ket1
out1 = apply_wick(full1)
out1.resolve()
final = AExpression(Ex=out1)
print("P_{ov}1_ia = ")
print(final)
print(final._print_einsum())

full1 = L*mid*ket2
out1 = apply_wick(full1)
out1.resolve()
final = AExpression(Ex=out1)
print("P_{ov}2_ia = ")
print(final)
print(final._print_einsum())

# vv block
operators = [FOperator(a, True), FOperator(b, False)]
pvv = Expression([Term(1, [], [Tensor([b, a], "")], operators, [])])

PT = commute(pvv, T)
PTT = commute(PT, T)
mid = pvv + PT + Fraction('1/2')*PTT
full = L*mid*ket1
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{vv}1_ab = ")
print(final)
print(final._print_einsum())

full = L*mid*ket2
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{vv}2_ab = ")
print(final)
print(final._print_einsum())

# oo block
operators = [FOperator(j, False), FOperator(i, True)]
poo = Expression([Term(-1, [], [Tensor([j, i], "")], operators, [])])

PT = commute(poo, T)
PTT = commute(PT, T)
mid = poo + PT + Fraction('1/2')*PTT
full = L*mid*ket1
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{oo}1_ij = ")
print(final)
print(final._print_einsum())

full = L*mid*ket2
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{oo}2_ij = ")
print(final)
print(final._print_einsum())


# vo block
operators = [FOperator(i, True), FOperator(a, False)]
pvo = Expression([Term(1, [], [Tensor([a, i], "")], operators, [])])

PT = commute(pvo, T)
PTT = commute(PT, T)
mid = pvo + PT + Fraction('1/2')*PTT
full = (mid + L*mid) * ket1
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{vo}1_ia = ")
print(final)
print(final._print_einsum())

full = (mid + L*mid) * ket2
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{vo}2_ia = ")
print(final)
print(final._print_einsum())
