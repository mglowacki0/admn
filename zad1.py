import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. INTERPOLACJA LAGRANGE'A
# ============================================================

def lagrange_value(x_nodes, y_nodes, x):
    """Oblicza wartosc wielomianu Lagrange'a w punkcie lub punktach x."""

    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x = np.asarray(x, dtype=float)

    result = np.zeros_like(x, dtype=float)
    n = len(x_nodes)

    # Budujemy kolejne skladniki y_i * L_i(x)
    for i in range(n):
        basis = np.ones_like(x, dtype=float)

        for k in range(n):
            if k != i:
                basis *= (
                    (x - x_nodes[k])
                    / (x_nodes[i] - x_nodes[k])
                )

        result += y_nodes[i] * basis

    return result


# ============================================================
# 2. INTERPOLACJA NEWTONA
# ============================================================

def newton_coefficients(x_nodes, y_nodes):
    """Wyznacza wspolczynniki postaci Newtona."""

    x_nodes = np.asarray(x_nodes, dtype=float)
    coef = np.asarray(y_nodes, dtype=float).copy()

    n = len(x_nodes)

    # Wyznaczanie ilorazow roznicowych
    for j in range(1, n):
        coef[j:n] = (
            coef[j:n] - coef[j - 1:n - 1]
        ) / (
            x_nodes[j:n] - x_nodes[0:n - j]
        )

    return coef


def newton_value(x_nodes, coef, x):
    """Oblicza wartosc wielomianu Newtona w punkcie lub punktach x."""

    x_nodes = np.asarray(x_nodes, dtype=float)
    coef = np.asarray(coef, dtype=float)
    x = np.asarray(x, dtype=float)

    result = np.zeros_like(x, dtype=float) + coef[-1]

    # Zagniezdzony schemat Hornera
    for k in range(len(coef) - 2, -1, -1):
        result = coef[k] + (x - x_nodes[k]) * result

    return result


# ============================================================
# 3. INTERPOLACJA NEVILLE'A
# ============================================================

def neville_value(x_nodes, y_nodes, x):
    """Oblicza wartosc wielomianu interpolacyjnego metoda Neville'a."""

    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    n = len(x_nodes)

    # Tabela Neville'a
    table = np.zeros((n, n), dtype=float)

    # Pierwsza kolumna zawiera wartosci y_i
    table[:, 0] = y_nodes

    # Obliczanie kolejnych kolumn
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (
                (x - x_nodes[i]) * table[i + 1, j - 1]
                - (x - x_nodes[i + j]) * table[i, j - 1]
            ) / (
                x_nodes[i + j] - x_nodes[i]
            )

    return table[0, n - 1]


# ============================================================
# 4. DANE
# ============================================================

x_nodes = np.array([-1.0, 0.0, 1.0, 2.0])
y_nodes = np.array([0.0, 1.0, 2.0, 9.0])


# ============================================================
# 5. WARTOSC W PUNKCIE x = 0.5
# ============================================================

x_test = 0.5

y_lagrange = lagrange_value(x_nodes, y_nodes, x_test)

coef = newton_coefficients(x_nodes, y_nodes)
y_newton = newton_value(x_nodes, coef, x_test)

y_neville = neville_value(x_nodes, y_nodes, x_test)

print("Wyniki dla x = 0.5")
print("------------------")
print("Lagrange:", y_lagrange)
print("Newton:  ", y_newton)
print("Neville: ", y_neville)

print()
print("Oczekiwana wartosc:")
print("P(0.5) =", 0.5**3 + 1)


# ============================================================
# 6. SPRAWDZENIE W WĘZŁACH
# ============================================================

print()
print("Wartosci w wezłach:")
print("-------------------")

print("Lagrange:")
print(lagrange_value(x_nodes, y_nodes, x_nodes))

print()
print("Newton:")
print(newton_value(x_nodes, coef, x_nodes))

print()
print("Neville:")

for x in x_nodes:
    print(neville_value(x_nodes, y_nodes, x))


# ============================================================
# 7. WSPOLCZYNNIKI NEWTONA
# ============================================================

print()
print("Wspolczynniki Newtona:")
print(coef)


# ============================================================
# 8. SIATKA DO WYKRESU
# ============================================================

x_plot = np.linspace(-1.25, 2.25, 500)

# Lagrange i Newton dzialaja od razu dla tablicy
y_lagrange_plot = lagrange_value(
    x_nodes,
    y_nodes,
    x_plot
)

y_newton_plot = newton_value(
    x_nodes,
    coef,
    x_plot
)

# Neville jest zaimplementowany dla pojedynczego x,
# dlatego obliczamy wartosc dla kazdego punktu siatki.
y_neville_plot = np.array([
    neville_value(x_nodes, y_nodes, x)
    for x in x_plot
])


# ============================================================
# 9. POROWNANIE WYNIKOW
# ============================================================

print()
print("Porownanie metod:")
print("-----------------")

print(
    "Lagrange = Newton:",
    np.allclose(y_lagrange_plot, y_newton_plot)
)

print(
    "Lagrange = Neville:",
    np.allclose(y_lagrange_plot, y_neville_plot)
)

print(
    "Newton = Neville:",
    np.allclose(y_newton_plot, y_neville_plot)
)


# ============================================================
# 10. WYKRES
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    x_plot,
    y_lagrange_plot,
    label="Lagrange"
)

plt.plot(
    x_plot,
    y_newton_plot,
    "--",
    label="Newton"
)

plt.plot(
    x_plot,
    y_neville_plot,
    ":",
    linewidth=3,
    label="Neville"
)

plt.scatter(
    x_nodes,
    y_nodes,
    color="red",
    zorder=5,
    label="wezly"
)

plt.xlabel("x")
plt.ylabel("P(x)")
plt.title("Interpolacja wielomianowa")
plt.grid(True)
plt.legend()

plt.show()


# ============================================================
# 11. OPCJONALNA KONTROLA ZA POMOCA SCIPY
# ============================================================

try:
    from scipy.interpolate import lagrange

    p_scipy = lagrange(x_nodes, y_nodes)

    print()
    print("Kontrola za pomoca SciPy:")
    print("-------------------------")

    print(
        "Wspolczynniki w kolejnosci malejacych poteg:"
    )
    print(p_scipy.coefficients)

    print(
        "P(0.5) obliczone przez SciPy:",
        p_scipy(0.5)
    )

except ImportError:
    print()
    print("SciPy nie jest zainstalowane.")