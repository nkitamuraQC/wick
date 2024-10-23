from fractions import Fraction
from wick.index import Idx
from wick.operator import FOperator, Tensor
from wick.expression import Term, Expression, AExpression
from wick.wick import apply_wick
from wick.convenience import E1, E2, commute, ketE1

index_key = {
    "oa": "ijklmno",
    "ob": "IJKLMNO",
    "va": "abcdefg",
    "vb": "ABCDEFG"
}

i = Idx(0, "oa")
a = Idx(0, "va")
j = Idx(1, "oa")
b = Idx(1, "va")

I = Idx(0, "ob")
A = Idx(0, "vb")
J = Idx(1, "ob")
B = Idx(1, "vb")

T1_a = E1("t", ["oa"], ["va"], index_key=index_key)
T1_b = E1("tb", ["O"], ["V"])
T2_aa = E2("T", ["oa", "oa"], ["va", "va"], index_key=index_key)
T2_bb = E2("tbb", ["O", "O"], ["V", "V"])
T2_ab = E2("U", ["oa", "ob"], ["va", "vb"], index_key=index_key)
T2_ba = E2("V", ["O", "o"], ["V", "v"])
#T = T1_a + T1_b + T2_aa + T2_bb + T2_ab
T = T1_a + T2_aa + T2_ab

L1_a = E1("l", ["va"], ["oa"], index_key=index_key)
L1_b = E1("lb", ["V"], ["O"])
L2_aa = E2("L", ["va", "va"], ["oa", "oa"], index_key=index_key)
L2_bb = E2("lbb", ["V", "V"], ["O", "O"])
L2_ab = E2("M", ["va", "vb"], ["oa", "ob"], index_key=index_key)
L2_ba = E2("N", ["V", "v"], ["O", "o"])
#L = L1_a + L1_b + L2_aa + L2_bb + L2_ab
L = L1_a + L2_aa + L2_ab 

R1_a = E1("r", ["oa"], ["va"], index_key=index_key)
R1_b = E1("rb", ["ob"], ["vb"])
R2_aa = E2("R", ["oa", "oa"], ["va", "va"], index_key=index_key)
R2_bb = E2("rbb", ["ob", "ob"], ["vb", "vb"])
R2_ab = E2("S", ["oa", "ob"], ["va", "vb"], index_key=index_key)
R2_ba = E2("rba", ["ob", "oa"], ["vb", "va"])
#R = R1_a + R1_b + R2_aa + R2_bb + R2_ab
R = R1_a + R2_aa + R2_ab

print()

# ov block
operators = [FOperator(a, True), FOperator(i, False)]
pvo = Expression([Term(1, [], [Tensor([i, a], "")], operators, [])])

PT = commute(pvo, T)
PTT = commute(PT, T)
mid = pvo + PT + Fraction('1/2')*PTT
full = L*mid*R
#full = L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
print("P_{ov} = ")
print(final._print_einsum())

# vv block
operators = [FOperator(a, True), FOperator(b, False)]
pvv = Expression([Term(1, [], [Tensor([b, a], "")], operators, [])])

PT = commute(pvv, T)
PTT = commute(PT, T)
mid = pvv + PT + Fraction('1/2')*PTT
full = L*mid*R
#full = L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{vv} = ")
print(final._print_einsum())

# oo block
operators = [FOperator(j, False), FOperator(i, True)]
poo = Expression([Term(-1, [], [Tensor([j, i], "")], operators, [])])

PT = commute(poo, T)
PTT = commute(PT, T)
mid = poo + PT + Fraction('1/2')*PTT
full = L*mid*R
#full = L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{oo} = ")
print(final._print_einsum())

# vo block
operators = [FOperator(i, True), FOperator(a, False)]
pvo = Expression([Term(1, [], [Tensor([a, i], "")], operators, [])])

PT = commute(pvo, T)
PTT = commute(PT, T)
mid = pvo + PT + Fraction('1/2')*PTT
full = (mid + L*mid)*R
#full = (mid + L*mid)
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{vo} = ")
print(final._print_einsum())
