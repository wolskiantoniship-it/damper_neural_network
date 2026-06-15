"""
diagnoza_odchylki.py
====================
Co robi (po ludzku):
Pomaga zrozumiec, SKAD bierze sie falowanie odchylki, ktore widac na
podgladzie etykiet. Sprawdza w szczegolnosci hipoteze:
"czy piki odchylki pokrywaja sie z ZAWRACANIEM tloka (predkosc ~ 0)?"
Jesli tak - falowanie to artefakt zbyt prostego wzorca C*v, a nie anomalia.

Pokazuje:
  1. HISTOGRAM biezacych odchylek - gdzie jest naturalna granica normy
     (gdzie histogram opada / ma "dolinke" - tam warto dac prog).
  2. Odchylke i predkosc w czasie dla jednego pliku NALOZONE na siebie -
     widac, czy piki odchylki padaja tam, gdzie predkosc przechodzi przez 0.
  3. Wykres odchylka vs |predkosc| - czy odchylka rosnie przy malej predkosci.

Nic nie zmienia. Czyta okna_dane.npz.

Uruchomienie:
    python diagnoza_odchylki.py
"""

import numpy as np
import matplotlib.pyplot as plt

PLIK = "okna_dane.npz"
C_WZORZEC = 0.00278
EPS_KN = 0.05    # taki sam jak w 02_etykietuj.py - dla odchylki wzglednej


def main():
    dane = np.load(PLIK, allow_pickle=True)
    X = dane["X"]
    id_pliku = dane["id_pliku"]

    # odchylka WZGLEDNA kazdego okna (taka jak w 02_etykietuj.py) + srednia predkosc
    odch = np.array([
        (np.abs(o[:, 0] - C_WZORZEC * o[:, 2]) /
         (np.abs(C_WZORZEC * o[:, 2]) + EPS_KN)).mean()
        for o in X
    ])
    v_sred = np.array([np.abs(o[:, 2]).mean() for o in X])   # srednia |predkosc| w oknie

    print(f"Odchylki WZGLEDNE: min {odch.min():.4f}, mediana {np.median(odch):.4f}, "
          f"max {odch.max():.4f}")
    print(f"Predkosc |v| w oknach: min {v_sred.min():.2f}, "
          f"mediana {np.median(v_sred):.2f}, max {v_sred.max():.2f} mm/s")

    # korelacja: czy mala predkosc idzie w parze z duza odchylka?
    korelacja = np.corrcoef(v_sred, odch)[0, 1]
    print(f"\nKorelacja |predkosc| <-> odchylka: {korelacja:.3f}")
    if korelacja < -0.3:
        print(">> UJEMNA: im mniejsza predkosc, tym wieksza odchylka.")
        print(">> To wspiera hipoteze: falowanie = artefakt wzorca przy zawracaniu tloka.")
    else:
        print(">> Brak silnej ujemnej korelacji - falowanie ma raczej inne zrodlo.")

    fig, osie = plt.subplots(2, 2, figsize=(13, 9))

    # 1. histogram odchylek - gdzie konczy sie norma
    ax = osie[0, 0]
    ax.hist(odch, bins=60, color="#888780")
    ax.set_title("Histogram wzglednych odchylek (gdzie dac prog?)")
    ax.set_xlabel("odchylka wzgledna")
    ax.set_ylabel("liczba okien")
    ax.axvline(np.median(odch), color="#378ADD", linestyle="--",
               label=f"mediana {np.median(odch):.3f}")
    ax.legend()

    # 2. odchylka i predkosc w czasie - jeden plik, nalozone
    ax = osie[0, 1]
    plik0 = np.unique(id_pliku)[0]
    m = id_pliku == plik0
    x_os = np.arange(int(m.sum()))
    ax.plot(x_os, odch[m], color="#D85A30", label="odchylka wzgledna")
    ax2 = ax.twinx()
    ax2.plot(x_os, v_sred[m], color="#378ADD", alpha=0.6, label="|predkosc| [mm/s]")
    ax.set_title(f"Plik #{plik0}: odchylka wzgledna vs predkosc w czasie")
    ax.set_xlabel("kolejne okno")
    ax.set_ylabel("odchylka wzgledna", color="#D85A30")
    ax2.set_ylabel("|predkosc| [mm/s]", color="#378ADD")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # 3. odchylka wzgledna vs predkosc - chmura punktow
    ax = osie[1, 0]
    ax.scatter(v_sred, odch, s=5, alpha=0.3, color="#534AB7")
    ax.set_title("Odchylka vs predkosc (kazde okno)")
    ax.set_xlabel("|predkosc| w oknie [mm/s]")
    ax.set_ylabel("odchylka wzgledna")

    # 4. histogram w skali log - lepiej widac "ogon" (rzadkie duze odchylki)
    ax = osie[1, 1]
    ax.hist(odch, bins=60, color="#888780")
    ax.set_yscale("log")
    ax.set_title("Histogram odchylek wzglednych (skala log - widac ogon)")
    ax.set_xlabel("odchylka wzgledna")
    ax.set_ylabel("liczba okien (log)")

    plt.tight_layout()
    plt.savefig("diagnoza_odchylki.png", dpi=120)
    print("\nWykresy zapisane do: diagnoza_odchylki.png")


if __name__ == "__main__":
    main()