"""
03_podglad_all.py
=================
Co robi (po ludzku):
Scala trzy osobne podglady w jeden skrypt i zapisuje WSZYSTKIE wyniki
(obrazki + tekstowe podsumowanie) do folderu wynikow.

Zawiera trzy czesci:
  1. PODGLAD OKIEN          - jak wygladaja dane okien (z okna_etykiety.npz)
  2. PODGLAD ETYKIET        - balans klas, predykcyjny sens (z okna_etykiety.npz)
  3. PODGLAD PROGU          - jak prog wplywa na balans klas (z okna_dane.npz)

Kazda czesc dziala niezaleznie - jak ktoregos pliku wejsciowego brakuje,
ta czesc jest pomijana z komunikatem, a reszta i tak sie wykona.

Stale i funkcje pochodza z parametry.py (jedno zrodlo prawdy).
Wszystkie wyniki laduja do folderu wynikow (parametry.FOLDER_WYNIKI).

Uruchomienie:
    python 03_podglad_all.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from parametry import (
    C_WZORZEC, HORYZONT_H,
    biezace_odchylki, etykiety_predykcyjne,
    sciezka_wyniku, Logger,
)

# ============================================================
# USTAWIENIA
# ============================================================

PLIK_ETYKIETY = "okna_etykiety.npz"   # dla czesci 1 i 2 (w folderze wynikow)
PLIK_DANE = "okna_dane.npz"           # dla czesci 3 (w folderze wynikow)

# czesc 1
ILE_OKIEN_NA_WYKRESIE = 10

# czesc 3 - progi do sprawdzenia (jak prog wplywa na balans klas)
PROGI = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]


# wspolny log na cala diagnoze
log = Logger("03_podglad_log.txt")


def wejscie(nazwa):
    """Pelna sciezka do pliku WEJSCIOWEGO (lezy w folderze wynikow)."""
    return sciezka_wyniku(nazwa)


# ============================================================
# CZESC 1 - PODGLAD OKIEN
# ============================================================

def czesc_podglad_okien():
    if not os.path.exists(wejscie(PLIK_ETYKIETY)):
        log(f"[CZESC 1] Pomijam - brak pliku {wejscie(PLIK_ETYKIETY)}")
        return

    log("=" * 60)
    log("CZESC 1: PODGLAD OKIEN")
    log("=" * 60)

    dane = np.load(wejscie(PLIK_ETYKIETY), allow_pickle=True)
    X = dane["X"]
    log(f"X - ksztalt: {X.shape}")
    log(f"  {X.shape[0]} okien, kazde {X.shape[1]} probek x {X.shape[2]} kolumny")
    log(f"  kolumny: [0]=sila[kN], [1]=przemieszczenie[mm], [2]=predkosc[mm/s]")

    if "id_pliku" in dane:
        idp = dane["id_pliku"]
        log(f"  unikalne pliki: {np.unique(idp)}")

    log("\nPierwsze okno, pierwsze 5 probek:")
    log("  sila      przem     predkosc")
    for wiersz in X[0][:5]:
        log(f"  {wiersz[0]:8.3f}  {wiersz[1]:8.3f}  {wiersz[2]:8.3f}")
    log("")

    n = min(ILE_OKIEN_NA_WYKRESIE, len(X))
    fig, osie = plt.subplots(n, 2, figsize=(11, 2.5 * n))
    if n == 1:
        osie = osie.reshape(1, 2)
    for i in range(n):
        okno = X[i]
        ax = osie[i, 0]
        ax.plot(okno[:, 0], label="sila [kN]")
        ax.plot(okno[:, 1], label="przem [mm]")
        ax.set_title(f"Okno {i} - przebieg w czasie")
        ax.set_xlabel("nr probki w oknie")
        ax.legend(fontsize=8)
        ax = osie[i, 1]
        ax.plot(okno[:, 2], okno[:, 0], ".-")
        ax.set_title(f"Okno {i} - sila vs predkosc")
        ax.set_xlabel("predkosc [mm/s]")
        ax.set_ylabel("sila [kN]")
    plt.tight_layout()
    plt.savefig(sciezka_wyniku("01_podglad_okien.png"), dpi=120)
    plt.close(fig)
    log(f"Zapisano: {sciezka_wyniku('01_podglad_okien.png')}\n")


# ============================================================
# CZESC 2 - PODGLAD ETYKIET
# ============================================================

def czesc_podglad_etykiet():
    if not os.path.exists(wejscie(PLIK_ETYKIETY)):
        log(f"[CZESC 2] Pomijam - brak pliku {wejscie(PLIK_ETYKIETY)}")
        return

    dane = np.load(wejscie(PLIK_ETYKIETY), allow_pickle=True)
    X = dane["X"]
    y = dane["y"]
    id_pliku = dane["id_pliku"]
    odchylki = dane["odchylki"] if "odchylki" in dane else None

    n = len(y)
    n_popr = int((y == 0).sum())
    n_odch = int((y == 1).sum())

    log("=" * 60)
    log("CZESC 2: PODGLAD ETYKIET (predykcyjnych)")
    log("=" * 60)
    log("Etykieta mowi: czy za H okien NADEJDZIE odchylka od normy.")
    log(f"Wszystkich okien z etykieta: {n}")
    log(f"  0 = spokojnie:        {n_popr}  ({100*n_popr/n:.1f}%)")
    log(f"  1 = nadchodzi:        {n_odch}  ({100*n_odch/n:.1f}%)")
    log("")
    log(f"{'plik':<10}{'okien':>8}{'spokojne':>10}{'nadchodzi':>11}{'%':>7}")
    log("-" * 50)
    for i in np.unique(id_pliku):
        m = id_pliku == i
        ile = int(m.sum())
        ile1 = int(y[m].sum())
        log(f"plik #{i:<5}{ile:>8}{ile-ile1:>10}{ile1:>11}{100*ile1/ile:>6.1f}%")
    log("")

    fig, osie = plt.subplots(2, 2, figsize=(13, 9))

    ax = osie[0, 0]
    ax.bar(["spokojne (0)", "nadchodzi (1)"], [n_popr, n_odch],
           color=["#1D9E75", "#D85A30"])
    ax.set_title("Liczba okien wg etykiety predykcyjnej")
    for idx, v in enumerate([n_popr, n_odch]):
        ax.text(idx, v, str(v), ha="center", va="bottom")

    ax = osie[0, 1]
    if odchylki is not None:
        ax.hist(odchylki[y == 0], bins=30, alpha=0.6, color="#1D9E75",
                label="spokojne (0)", density=True)
        ax.hist(odchylki[y == 1], bins=30, alpha=0.6, color="#D85A30",
                label="nadchodzi (1)", density=True)
        ax.set_title("Biezaca odchylka vs przyszla etykieta")
        ax.set_xlabel("biezaca odchylka okna")
        ax.set_ylabel("gestosc")
        ax.legend()
        ax.text(0.5, 0.95, "nakladanie sie = etykieta NIE wynika\nwprost z biezacej wartosci (dobrze!)",
                transform=ax.transAxes, fontsize=8, va="top", ha="center", color="#444441")
    else:
        ax.axis("off")

    ax = osie[1, 0]
    pliki = np.unique(id_pliku)
    proc = [100 * y[id_pliku == i].sum() / (id_pliku == i).sum() for i in pliki]
    ax.bar([str(i) for i in pliki], proc, color="#378ADD")
    ax.set_title("% okien zapowiadajacych odchylke (per plik)")
    ax.set_xlabel("numer pliku")
    ax.set_ylabel("% etykiet = 1")
    ax.tick_params(axis="x", labelsize=7)

    ax = osie[1, 1]
    plik_demo = pliki[int(np.argmax(proc))]
    m = id_pliku == plik_demo
    if odchylki is not None:
        od_plik = odchylki[m]
        y_plik = y[m]
        x_os = np.arange(len(od_plik))
        ax.plot(x_os, od_plik, color="#5F5E5A", label="biezaca odchylka")
        ax.scatter(x_os[y_plik == 1], od_plik[y_plik == 1], s=15,
                   color="#D85A30", label="etykieta=1 (nadchodzi)", zorder=3)
        ax.set_title(f"Plik #{plik_demo}: odchylka w czasie + zapowiedzi")
        ax.set_xlabel("kolejne okno w pliku")
        ax.set_ylabel("biezaca odchylka")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(sciezka_wyniku("02_podglad_etykiet.png"), dpi=120)
    plt.close(fig)
    log(f"Zapisano: {sciezka_wyniku('02_podglad_etykiet.png')}\n")


# ============================================================
# CZESC 3 - PODGLAD PROGU
# ============================================================

def czesc_podglad_progu():
    if not os.path.exists(wejscie(PLIK_DANE)):
        log(f"[CZESC 3] Pomijam - brak pliku {wejscie(PLIK_DANE)}")
        return

    dane = np.load(wejscie(PLIK_DANE), allow_pickle=True)
    X = dane["X"]
    id_pliku = dane["id_pliku"]

    odch = biezace_odchylki(X, C_WZORZEC)
    log("=" * 60)
    log("CZESC 3: PODGLAD PROGU")
    log("=" * 60)
    log(f"Odchylki wzgledne: mediana {np.median(odch):.3f}, max {odch.max():.3f}")
    log(f"Horyzont H = {HORYZONT_H}\n")
    log(f"{'prog':>6} | {'% klasy 1':>10} | {'poprawne':>9} | {'odchylki':>9}")
    log("-" * 45)

    procenty = []
    for p in PROGI:
        y = etykiety_predykcyjne(odch, id_pliku, HORYZONT_H, p)
        m = y >= 0
        proc = 100 * y[m].sum() / m.sum() if m.sum() else 0
        procenty.append(proc)
        log(f"{p:>6.2f} | {proc:>9.1f}% | {int((y[m]==0).sum()):>9} | {int(y[m].sum()):>9}")

    log("\nWSKAZOWKA: szukaj progu, gdzie klasa 1 to ~10-40%.")
    log("")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(PROGI, procenty, "o-", color="#D85A30")
    ax1.axhspan(10, 40, alpha=0.15, color="#1D9E75", label="strefa sensowna (10-40%)")
    ax1.set_xlabel("prog (wzgledny)")
    ax1.set_ylabel("% okien w klasie 1")
    ax1.set_title("Jak prog wplywa na balans klas")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.hist(odch, bins=60, color="#888780")
    for p in PROGI:
        ax2.axvline(p, color="#378ADD", alpha=0.4, linewidth=0.8)
    ax2.axvline(np.median(odch), color="#D85A30", linestyle="--",
                label=f"mediana {np.median(odch):.2f}")
    ax2.set_xlabel("odchylka wzgledna")
    ax2.set_ylabel("liczba okien")
    ax2.set_title("Rozklad odchylek + sprawdzane progi")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(sciezka_wyniku("03_podglad_progu.png"), dpi=120)
    plt.close(fig)
    log(f"Zapisano: {sciezka_wyniku('03_podglad_progu.png')}\n")


# ============================================================
# GLOWNA CZESC
# ============================================================

def main():
    log(f"Wszystkie wyniki laduja do folderu: {sciezka_wyniku('')}\n")

    czesc_podglad_okien()
    czesc_podglad_etykiet()
    czesc_podglad_progu()

    log.zapisz()
    print(f"Gotowe - zajrzyj do folderu wynikow.")


if __name__ == "__main__":
    main()
