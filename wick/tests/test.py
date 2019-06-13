from wick.expression import *
from wick.orbital_space import OrbitalSpace
from wick.wick import apply_wick

o1 = OrbitalSpace("o1")
v1 = OrbitalSpace("v1")

s = 0.5
sums = []
tensors = []
operators = []
deltas = []

indices = ["a","b"]
indices2 = ["i","j"]
spaces = [v1, v1]
spaces2 = [o1, o1]
bks = [True, False]
sums.append(Sigma(indices[0], v1))
sums.append(Sigma(indices[1], v1))
sums.append(Sigma("i", o1))
tensors.append(Tensor(indices2, spaces2, bks, "f"))
tensors.append(Tensor(indices, spaces, bks, "f"))
operators.append(Operator("a", v1, False))
operators.append(Operator("b", v1, True))
operators.append(Operator("i", o1, True))
operators.append(Operator("j", o1, False))

t = Term(s,sums, tensors, operators, deltas)

e = Expression([t])
print(e.print_str())
x = apply_wick(e)
print(x.print_str())
x.resolve()
print(x.print_str())
