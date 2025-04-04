import numpy as np

# Parametry algorytmu
c = 0.01  # Współczynnik uczenia
epsilon = 0.0001  # Kryterium stopu
beta = 2.0  # Współczynnik w funkcji aktywacji
max_iter = 10000  # Maksymalna liczba iteracji

# Funkcja aktywacji - Sigmoid
def f(x):
    return 1 / (1 + np.exp(-beta * x))

# Pochodna Sigmoida
def df(x):
    return beta * f(x) * (1 - f(x))

# Wejścia (4 próbki XOR) + bias
u_p = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Oczekiwane wyjścia XOR
z_p = np.array([0, 1, 1, 0])

# Inicjalizacja wag warstw
np.random.seed(42)  # Powtarzalność wyników
w_old = np.random.uniform(-1, 1, (2, 3))  # Wagi wejście -> warstwa ukryta (2x3)
s_old = np.random.uniform(-1, 1, (3,))    # Wagi warstwa ukryta -> wyjście (3,)

t = 0  # Licznik iteracji

# Obliczenia początkowe
for p in range(4):  # Dla każdej próbki XOR
    x_p = f(np.dot(w_old, u_p[p]))  # Obliczanie sygnału w warstwie ukrytej
    y_p = f(np.dot(s_old, x_p))  # Wyjście końcowe sieci
    print(f"t = {t}, y({p+1}) = {y_p:.4f}")

# Pętla uczenia
while t < max_iter:
    t += 1
    p = np.random.randint(0, 4)  # Losowanie próbki
    
    # Przepuszczenie sygnału przez sieć
    x_p = f(np.dot(w_old, u_p[p]))  
    y_p = f(np.dot(s_old, x_p))  

    # Obliczenie gradientów
    DEp_s = (y_p - z_p[p]) * df(np.dot(s_old, x_p)) * x_p  
    DEp_w = np.outer((y_p - z_p[p]) * df(np.dot(s_old, x_p)) * s_old, df(np.dot(w_old, u_p[p])) * u_p[p])

    # Aktualizacja wag
    s_new = s_old - c * DEp_s
    w_new = w_old - c * DEp_w

    # Warunek stopu
    if np.max(np.abs(s_new - s_old)) < epsilon and np.max(np.abs(w_new - w_old)) < epsilon:
        break

    # Aktualizacja wag do kolejnej iteracji
    s_old, w_old = s_new, w_new

# Wyświetlenie wyników końcowych
print(f"\nLiczba iteracji: {t}")

for p in range(4):
    x_p = f(np.dot(w_old, u_p[p]))
    y_p = f(np.dot(s_old, x_p))
    print(f"y({p+1}) = {y_p:.4f} (oczekiwane: {z_p[p]})")
