import numpy as np

c = 0.1
epsilon = 0.00001
beta = 1.0
N = 1

x = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])
z = np.array([0, 0, 0, 1])

s_old = np.random.uniform(-N, N, 3)

def f(x):
    return 1 / (1 + np.exp(-beta * x))

def df(x):
    return beta * f(x) * (1 - f(x))

def compute_gradient(s_old):
    gradient = np.zeros(3)
    for p in range(4):
        net = np.dot(s_old, x[p])
        y_p = f(net)
        error = y_p - z[p]
        gradient += error * df(net) * x[p]
    return gradient

t = 0
print(f"Iteracja {t}, wagi: {s_old}")
while True:
    s_new = s_old - c * compute_gradient(s_old)
    if np.max(np.abs(s_new - s_old)) <= epsilon:
        break
    
    s_old = s_new
    t += 1

print(f"Iteracja {t}, finalne wagi: {s_old}")

for p in range(4):
    y_p = f(np.dot(s_old, x[p]))
    print(f"Wejście: {x[p][:2]}, Wyjście: {y_p:.4f}, Oczekiwane: {z[p]}")
