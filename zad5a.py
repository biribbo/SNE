import numpy as np

c = 0.6
epsilon = 0.000001
beta = 1.0
N = 1.0
max_iter = 10000

def f(x):
    return 1 / (1 + np.exp(-beta * x))

def df(x):
    return beta * f(x) * (1 - f(x))

u = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
z = np.array([[0], [1], [1], [0]])

# Inicjalizacja wag
np.random.seed(42)
w_old = np.random.uniform(-N, N, (3, 2))
s_old = np.random.uniform(-N, N, (3, 1))

t = 0
while t < max_iter:
    net_hidden = np.dot(u, w_old)
    x = f(net_hidden)
    
    x = np.hstack([x, np.ones((x.shape[0], 1))])

    net_output = np.dot(x, s_old)
    y = f(net_output)

    error = y - z
    
    delta_s = error * df(net_output)
    de_s = np.dot(x.T, delta_s)

    delta_hidden = np.dot(delta_s, s_old.T)[:, :-1] * df(net_hidden)  # (4,2)
    de_w = np.dot(u.T, delta_hidden)  # (3,4) * (4,2) = (3,2)

    s_new = s_old - c * de_s
    w_new = w_old - c * de_w

    if np.max(np.abs(s_new - s_old)) < epsilon and np.max(np.abs(w_new - w_old)) < epsilon:
        break

    s_old, w_old = s_new, w_new
    t += 1

print(f"Liczba iteracji: {t}")
print("y po końcu iteracji:", f(np.dot(np.hstack([f(np.dot(u, w_old)), np.ones((u.shape[0], 1))]), s_old)).flatten())
