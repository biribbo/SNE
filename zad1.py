import numpy as np
from itertools import product

def step_function(weights, inputs):
    return int(np.dot(weights, inputs) >= 0)

def NOT(u):
    weights = np.array([-1.5, 1])
    inputs = np.array([u, 1])
    return step_function(weights, inputs)

def AND(u1, u2):
    weights = np.array([1, 1, -1.25])
    inputs = np.array([u1, u2, 1])
    return step_function(weights, inputs)

def NAND(u1, u2):
    weights = np.array([-1, -1, 1.25])
    inputs = np.array([u1, u2, 1])
    return step_function(weights, inputs)

def OR(u1, u2):
    weights = np.array([1.5, 1.5, -1])
    inputs = np.array([u1, u2, 1])
    return step_function(weights, inputs)

print("NOT:")
for u1 in [0, 1]:
    print(f"NOT({u1}) = {NOT(u1)}")

print("\nAND:")
for u1, u2 in product([0, 1], repeat=2):
    print(f"AND({u1}, {u2}) = {AND(u1, u2)}")

print("\nNAND:")
for u1, u2 in product([0, 1], repeat=2):
    print(f"NAND({u1}, {u2}) = {NAND(u1, u2)}")

print("\nOR:")
for u1, u2 in product([0, 1], repeat=2):
    print(f"OR({u1}, {u2}) = {OR(u1, u2)}")
