"""
podglad_all.py
==============
Co robi (po ludzku):
Scala trzy osobne podglady w jeden skrypt i zapisuje WSZYSTKIE wyniki
(obrazki + tekstowe podsumowanie) do osobnego katalogu.

Zawiera trzy czesci:
  1. PODGLAD OKIEN          - jak wygladaja dane okien (z okna_etykiety.npz)
  2. PODGLAD ETYKIET        - balans klas, predykcyjny sens (z okna_etykiety.npz)
  3. PODGLAD PROGU          - jak prog wplywa na balans klas (z okna_dane.npz)

Kazda czesc dziala niezaleznie - jak ktoregos pliku wejsciowego brakuje,
ta czesc jest pomijana z komunikatem, a reszta i tak sie wykona.

Wszystko laduje do katalogu KATALOG_WYNIKOW (domyslnie 'wyniki_podgladu').

Uruchomienie:
    python podglad_all.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USTAWIENIA
# ============================================================
KATALOG_WYNIKOW = "wyniki_podgladu"

PLIK_ETYKIETY = "okna_etykiety.npz"   # dla czesci 1 i 2
PLIK_DANE = "okna_dane.npz"           # dla czesci 3

# czesc 1
ILE_OKIEN_NA_WYKRESIE = 10

# czesc 3 (musi pasowac do 02_etykietuj.py)
C_WZORZEC = 0.00278
EPS_KN = 0.05
HORYZONT_H = 10
PROGI = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]


# bufor na tekst - zapiszemy go tez do pliku, nie tylko na ekran
_raport = []

def loguj(tekst=""):
    """Wypisuje na ekran I zapamietuje do raportu tekstowego."""
    print(tekst)
    _raport.append(tekst)


def sciezka(nazwa):
    """Pelna sciezka do pliku w katalogu wynikow."""
    return os.path.join(KATALOG_WYNIKOW, nazwa)


# ============================================================
# WSPOLNE FUNKCJE (dla czesci 3)
# ============================================================

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


# ============================================================
# CZESC 1 - PODGLAD OKIEN
# ============================================================

def czesc_podglad_okien():
    if not os.path.exists(PLIK_ETYKIETY):
        loguj(f"[CZESC 1] Pomijam - brak pliku {PLIK_ETYKIETY}")
        return

    loguj("=" * 60)
    loguj("CZESC 1: PODGLAD OKIEN")
    loguj("=" * 60)

    dane = np.load(PLIK_ETYKIETY, allow_pickle=True)
    X = dane["X"]
    loguj(f"X - ksztalt: {X.shape}")
    loguj(f"  {X.shape[0]} okien, kazde {X.shape[1]} probek x {X.shape[2]} kolumny")
    loguj(f"  kolumny: [0]=sila[kN], [1]=przemieszczenie[mm], [2]=predkosc[mm/s]")

    if "id_pliku" in dane:
        idp = dane["id_pliku"]
        loguj(f"  unikalne pliki: {np.unique(idp)}")

    loguj("\nPierwsze okno, pierwsze 5 probek:")
    loguj("  sila      przem     predkosc")
    for wiersz in X[0][:5]:
        loguj(f"  {wiersz[0]:8.3f}  {wiersz[1]:8.3f}  {wiersz[2]:8.3f}")
    loguj("")

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
    plt.savefig(sciezka("01_podglad_okien.png"), dpi=120)
    plt.close(fig)
    loguj(f"Zapisano: {sciezka('01_podglad_okien.png')}\n")


# ============================================================
# CZESC 2 - PODGLAD ETYKIET
# ============================================================

def czesc_podglad_etykiet():
    if not os.path.exists(PLIK_ETYKIETY):
        loguj(f"[CZESC 2] Pomijam - brak pliku {PLIK_ETYKIETY}")
        return

    dane = np.load(PLIK_ETYKIETY, allow_pickle=True)
    X = dane["X"]
    y = dane["y"]
    id_pliku = dane["id_pliku"]
    odchylki = dane["odchylki"] if "odchylki" in dane else None

    n = len(y)
    n_popr = int((y == 0).sum())
    n_odch = int((y == 1).sum())

    loguj("=" * 60)
    loguj("CZESC 2: PODGLAD ETYKIET (predykcyjnych)")
    loguj("=" * 60)
    loguj("Etykieta mowi: czy za H okien NADEJDZIE odchylka od normy.")
    loguj(f"Wszystkich okien z etykieta: {n}")
    loguj(f"  0 = spokojnie:        {n_popr}  ({100*n_popr/n:.1f}%)")
    loguj(f"  1 = nadchodzi:        {n_odch}  ({100*n_odch/n:.1f}%)")
    loguj("")
    loguj(f"{'plik':<10}{'okien':>8}{'spokojne':>10}{'nadchodzi':>11}{'%':>7}")
    loguj("-" * 50)
    for i in np.unique(id_pliku):
        m = id_pliku == i
        ile = int(m.sum())
        ile1 = int(y[m].sum())
        loguj(f"plik #{i:<5}{ile:>8}{ile-ile1:>10}{ile1:>11}{100*ile1/ile:>6.1f}%")
    loguj("")

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
    plt.savefig(sciezka("02_podglad_etykiet.png"), dpi=120)
    plt.close(fig)
    loguj(f"Zapisano: {sciezka('02_podglad_etykiet.png')}\n")


# ============================================================
# CZESC 3 - PODGLAD PROGU
# ============================================================

def czesc_podglad_progu():
    if not os.path.exists(PLIK_DANE):
        loguj(f"[CZESC 3] Pomijam - brak pliku {PLIK_DANE}")
        return

    dane = np.load(PLIK_DANE, allow_pickle=True)
    X = dane["X"]
    id_pliku = dane["id_pliku"]

    odch = biezace_odchylki(X, C_WZORZEC)
    loguj("=" * 60)
    loguj("CZESC 3: PODGLAD PROGU")
    loguj("=" * 60)
    loguj(f"Odchylki wzgledne: mediana {np.median(odch):.3f}, max {odch.max():.3f}")
    loguj(f"Horyzont H = {HORYZONT_H}\n")
    loguj(f"{'prog':>6} | {'% klasy 1':>10} | {'poprawne':>9} | {'odchylki':>9}")
    loguj("-" * 45)

    procenty = []
    for p in PROGI:
        y = etykiety_predykcyjne(odch, id_pliku, HORYZONT_H, p)
        m = y >= 0
        proc = 100 * y[m].sum() / m.sum() if m.sum() else 0
        procenty.append(proc)
        loguj(f"{p:>6.2f} | {proc:>9.1f}% | {int((y[m]==0).sum()):>9} | {int(y[m].sum()):>9}")

    loguj("\nWSKAZOWKA: szukaj progu, gdzie klasa 1 to ~10-40%.")
    loguj("")

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
    plt.savefig(sciezka("03_podglad_progu.png"), dpi=120)
    plt.close(fig)
    loguj(f"Zapisano: {sciezka('03_podglad_progu.png')}\n")


# ============================================================
# GLOWNA CZESC
# ============================================================

def main():
    os.makedirs(KATALOG_WYNIKOW, exist_ok=True)
    loguj(f"Wszystkie wyniki ladnie do katalogu: {KATALOG_WYNIKOW}/\n")

    czesc_podglad_okien()
    czesc_podglad_etykiet()
    czesc_podglad_progu()

    # zapisz cale tekstowe podsumowanie do pliku
    with open(sciezka("podsumowanie.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_raport))
    print(f"\nTekstowe podsumowanie zapisane do: {sciezka('podsumowanie.txt')}")
    print(f"Gotowe - zajrzyj do katalogu '{KATALOG_WYNIKOW}'.")


if __name__ == "__main__":
    main()
