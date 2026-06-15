"""
podglad_etykiet.py   (wersja dla etykiet PREDYKCYJNYCH)
========================================================
Co robi (po ludzku):
Pokazuje obraz danych po NOWYM etykietowaniu predykcyjnym (z 02_etykietuj.py).

WAZNE rozroznienie, ktore widac na wykresach:
  - 'odchylki' w pliku = BIEZACA odchylka okna (to, co okno pokazuje TERAZ),
  - 'y' (etykieta)     = czy za H okien NADEJDZIE odchylka (PRZYSZLOSC).
Czyli okno moze byc teraz spokojne (mala biezaca odchylka), a i tak miec
etykiete 1, bo zapowiada nadchodzacy problem. Wykresy to pokazuja.

Pokazuje:
  1. ile okien oznaczono 0 / 1,
  2. rozbicie per plik,
  3. wykresy: liczba etykiet, biezaca odchylka w rozbiciu na etykiete,
     przebieg odchylki w czasie dla przykladowego pliku (z zaznaczeniem,
     ktore okna zapowiadaja przyszla odchylke).

Uruchomienie:
    python podglad_etykiet.py
"""

import numpy as np
import matplotlib.pyplot as plt

PLIK = "okna_etykiety.npz"


def main():
    dane = np.load(PLIK, allow_pickle=True)
    X = dane["X"]
    y = dane["y"]
    id_pliku = dane["id_pliku"]
    odchylki = dane["odchylki"] if "odchylki" in dane else None

    n = len(y)
    n_popr = int((y == 0).sum())
    n_odch = int((y == 1).sum())

    # --- 1. PODSUMOWANIE ---
    print("=" * 60)
    print("PODSUMOWANIE - ETYKIETY PREDYKCYJNE")
    print("=" * 60)
    print("Etykieta mowi: czy za H okien NADEJDZIE odchylka od normy.")
    print(f"Wszystkich okien z etykieta: {n}")
    print(f"  0 = spokojnie (nic nie nadchodzi):  {n_popr}  ({100*n_popr/n:.1f}%)")
    print(f"  1 = nadchodzi odchylka:             {n_odch}  ({100*n_odch/n:.1f}%)")
    print()

    # --- 2. PER PLIK ---
    print("=" * 60)
    print("ROZBICIE PER PLIK")
    print("=" * 60)
    print(f"{'plik':<10}{'okien':>8}{'spokojne':>10}{'nadchodzi':>11}{'%':>7}")
    print("-" * 60)
    for i in np.unique(id_pliku):
        m = id_pliku == i
        ile = int(m.sum())
        ile1 = int(y[m].sum())
        ile0 = ile - ile1
        print(f"plik #{i:<5}{ile:>8}{ile0:>10}{ile1:>11}{100*ile1/ile:>6.1f}%")
    print()

    # --- 3. WYKRESY ---
    fig, osie = plt.subplots(2, 2, figsize=(13, 9))

    # 3a. slupki: ile etykiet 0 vs 1
    ax = osie[0, 0]
    ax.bar(["spokojne (0)", "nadchodzi (1)"], [n_popr, n_odch],
           color=["#1D9E75", "#D85A30"])
    ax.set_title("Liczba okien wg etykiety predykcyjnej")
    for idx, v in enumerate([n_popr, n_odch]):
        ax.text(idx, v, str(v), ha="center", va="bottom")

    # 3b. biezaca odchylka w rozbiciu na etykiete
    # KLUCZOWE: pokazuje, ze etykieta NIE jest prosta funkcja biezacej odchylki
    # (gdyby byla - rozklady 0 i 1 bylyby idealnie rozdzielone)
    ax = osie[0, 1]
    if odchylki is not None:
        ax.hist(odchylki[y == 0], bins=30, alpha=0.6, color="#1D9E75",
                label="spokojne (0)", density=True)
        ax.hist(odchylki[y == 1], bins=30, alpha=0.6, color="#D85A30",
                label="nadchodzi (1)", density=True)
        ax.set_title("Biezaca odchylka vs przyszla etykieta")
        ax.set_xlabel("biezaca odchylka okna [kN]")
        ax.set_ylabel("gestosc")
        ax.legend()
        # podpowiedz interpretacyjna na wykresie
        ax.text(0.5, 0.95, "nakladanie sie = etykieta NIE wynika\nwprost z biezacej wartosci (dobrze!)",
                transform=ax.transAxes, fontsize=8, va="top", ha="center",
                color="#444441")
    else:
        ax.axis("off")

    # 3c. % nadchodzacych odchylek per plik
    ax = osie[1, 0]
    pliki = np.unique(id_pliku)
    proc = [100 * y[id_pliku == i].sum() / (id_pliku == i).sum() for i in pliki]
    ax.bar([str(i) for i in pliki], proc, color="#378ADD")
    ax.set_title("% okien zapowiadajacych odchylke (per plik)")
    ax.set_xlabel("numer pliku")
    ax.set_ylabel("% etykiet = 1")
    ax.tick_params(axis="x", labelsize=7)

    # 3d. przebieg w czasie dla jednego pliku: biezaca odchylka + gdzie etykieta=1
    # pokazuje predykcyjny sens: etykieta=1 pojawia sie PRZED wzrostem odchylki
    ax = osie[1, 1]
    # wybierz plik z najwieksza liczba etykiet 1 (najciekawszy do pokazania)
    plik_demo = pliki[int(np.argmax(proc))]
    m = id_pliku == plik_demo
    if odchylki is not None:
        od_plik = odchylki[m]
        y_plik = y[m]
        x_os = np.arange(len(od_plik))
        ax.plot(x_os, od_plik, color="#5F5E5A", label="biezaca odchylka")
        # zaznacz okna z etykieta 1 (zapowiadaja przyszla odchylke)
        ax.scatter(x_os[y_plik == 1], od_plik[y_plik == 1], s=15,
                   color="#D85A30", label="etykieta=1 (nadchodzi)", zorder=3)
        ax.set_title(f"Plik #{plik_demo}: odchylka w czasie + zapowiedzi")
        ax.set_xlabel("kolejne okno w pliku")
        ax.set_ylabel("biezaca odchylka [kN]")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("podglad_etykiet.png", dpi=120)
    print("Wykresy zapisane do: podglad_etykiet.png")


if __name__ == "__main__":
    main()