"""
podglad_progu.py
================
Co robi (po ludzku):
Pomaga WYBRAC prog do etykietowania - na podstawie liczb, nie na oko.
Dla siatki progow pokazuje, ile okien wpadnie do klasy 1 ("nadchodzi odchylka")
po etykietowaniu predykcyjnym z horyzontem H.

Po co: prog to Twoja decyzja, ktora uzasadnisz w artykule. Ten skrypt
pokazuje konsekwencje kazdego wyboru - zeby nie wyszlo 92% (prog za niski)
ani 1% (prog za wysoki), tylko sensowny, wyrazisty odsetek klasy 1.

Uzywa DOKLADNIE tej samej odchylki wzglednej co 02_etykietuj.py.

Czyta okna_dane.npz. Nic nie zapisuje (poza wykresem).

Uruchomienie:
    python podglad_progu.py
"""

import numpy as np
import matplotlib.pyplot as plt

PLIK = "okna_dane.npz"
C_WZORZEC = 0.00278
EPS_KN = 0.05
HORYZONT_H = 10           # taki sam jak w 02_etykietuj.py

# siatka progow do sprawdzenia (wzgledne, bezwymiarowe)
PROGI = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]


def biezace_odchylki(X, C):
    wynik = []
    for o in X:
        F, v = o[:, 0], o[:, 2]
        wzor = C * v
        wynik.append((np.abs(F - wzor) / (np.abs(wzor) + EPS_KN)).mean())
    return np.array(wynik)


def etykiety_predykcyjne(odchylki, id_pliku, H, prog):
    n = len(odchylki)
    y = np.full(n, -1, dtype=int)
    for i in range(n):
        j = i + H
        if j < n and id_pliku[j] == id_pliku[i]:
            y[i] = 1 if odchylki[j] > prog else 0
    return y


def main():
    dane = np.load(PLIK, allow_pickle=True)
    X = dane["X"]
    id_pliku = dane["id_pliku"]

    odch = biezace_odchylki(X, C_WZORZEC)
    print(f"Odchylki wzgledne: mediana {np.median(odch):.3f}, max {odch.max():.3f}")
    print(f"Horyzont H = {HORYZONT_H}\n")

    print(f"{'prog':>6} | {'% klasy 1':>10} | {'poprawne':>9} | {'odchylki':>9}")
    print("-" * 45)

    procenty = []
    for p in PROGI:
        y = etykiety_predykcyjne(odch, id_pliku, HORYZONT_H, p)
        m = y >= 0
        proc = 100 * y[m].sum() / m.sum() if m.sum() else 0
        procenty.append(proc)
        print(f"{p:>6.2f} | {proc:>9.1f}% | {int((y[m]==0).sum()):>9} | {int(y[m].sum()):>9}")

    print("\nWSKAZOWKA: szukaj progu, gdzie klasa 1 to ~10-40%.")
    print("Za malo (<5%) - siec nie ma sie na czym uczyc.")
    print("Za duzo (>70%) - klasa 1 przestaje znaczyc 'wyjatek'.")

    # --- wykres: % klasy 1 vs prog ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(PROGI, procenty, "o-", color="#D85A30")
    ax1.axhspan(10, 40, alpha=0.15, color="#1D9E75", label="strefa sensowna (10-40%)")
    ax1.set_xlabel("prog (wzgledny)")
    ax1.set_ylabel("% okien w klasie 1")
    ax1.set_title("Jak prog wplywa na balans klas")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # histogram odchylek z naniesionymi progami - widac gdzie ciana rozkladu
    ax2.hist(odch, bins=60, color="#888780")
    for p in PROGI:
        ax2.axvline(p, color="#378ADD", alpha=0.4, linewidth=0.8)
    ax2.axvline(np.median(odch), color="#D85A30", linestyle="--",
                label=f"mediana {np.median(odch):.2f}")
    ax2.set_xlabel("odchylka wzgledna")
    ax2.set_ylabel("liczba okien")
    ax2.set_title("Rozklad odchylek + sprawdzane progi (niebieskie)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("podglad_progu.png", dpi=120)
    print("\nWykres zapisany do: podglad_progu.png")


if __name__ == "__main__":
    main()
