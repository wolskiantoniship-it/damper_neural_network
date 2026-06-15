"""
02_etykietuj.py   (wersja PREDYKCYJNA - Pomysl 1)
==================================================
Co robi (po ludzku):
Nadaje kazdemu oknu etykiete 0/1, ale etykieta dotyczy PRZYSZLOSCI:
  etykieta okna n = 1, jesli za H okien (czyli w oknie n+H) sila odjedzie
  od normy bardziej niz prog.
Sieci pokazujemy okno n (terazniejszosc), a ona ma przewidziec, czy nadchodzi
odchylka. To jest "wczesne ostrzeganie" - przewidywanie narastajacego trendu.

DLACZEGO nie ma sciagi:
Etykieta opisuje okno n+H, ktorego siec NIE widzi. Zeby trafic, musi wylapac
w biezacym oknie zapowiedz trendu, a nie odtworzyc regule.

NORMA (wzorzec):
Sila oczekiwana = C * predkosc (krzywa wzorcowa z modelu liniowego).
Biezaca odchylka okna = jak bardzo zmierzona sila odbiega od C*v.

WAZNE: okna n i n+H musza byc z TEGO SAMEGO pliku. Okna przy koncu pliku,
dla ktorych nie ma n+H, dostaja etykiete -1 i sa pomijane (zapisujemy maske).

Uruchomienie:
    python 02_etykietuj.py
"""

import numpy as np

PLIK_WEJSCIOWY = "okna_dane.npz"
PLIK_WYJSCIOWY = "okna_etykiety.npz"

# --- wspolczynnik wzorca (z modelu liniowego F = C*v) ---
C_WZORZEC = 0.00278

# --- HORYZONT predykcji: ile okien w przod patrzy etykieta ---
HORYZONT_H = 10

# === PROG odchylki WZGLEDNEJ (bezwymiarowy) - Twoja decyzja ===
# odchylka jest teraz WZGLEDNA (ulamek odstepstwa od wzorca), wiec prog
# tez jest bezwymiarowy: 0.30 = "sila odbiega srednio o 30% od oczekiwanej".
# Dobierz patrzac na rozklad (diagnoza/podglad).
PROG_WZGLEDNY = 4

# epsilon chroni przed dzieleniem przez ~0, gdy tlok zawraca (v~0 -> C*v~0).
# Jednostka: kN. Dobrany jako maly ulamek typowej sily.
EPS_KN = 0.05


def biezace_odchylki(X, C):
    """
    Dla kazdego okna: srednie WZGLEDNE odstepstwo sily od wzorca.
    = |zmierzona - C*v| / (|C*v| + eps)
    Dzielenie przez wartosc wzorcowa usuwa zaleznosc od bezwzglednej predkosci:
    20% odchylki znaczy to samo przy wolnym i szybkim ruchu tloka.
    eps zapobiega wybuchowi przy zawracaniu tloka (v~0).
    """
    wynik = []
    for o in X:
        F = o[:, 0]
        v = o[:, 2]
        wzorzec = C * v
        wzgledne = np.abs(F - wzorzec) / (np.abs(wzorzec) + EPS_KN)
        wynik.append(wzgledne.mean())
    return np.array(wynik)


def etykiety_predykcyjne(odchylki, id_pliku, H, prog):
    """
    Etykieta okna n = 1, jesli odchylka okna (n+H) > prog.
    Tylko gdy n i n+H sa z tego samego pliku; inaczej -1 (pomijamy).
    """
    n = len(odchylki)
    y = np.full(n, -1, dtype=int)
    for i in range(n):
        j = i + H
        if j < n and id_pliku[j] == id_pliku[i]:
            y[i] = 1 if odchylki[j] > prog else 0
    return y


def main():
    dane = np.load(PLIK_WEJSCIOWY, allow_pickle=True)
    X = dane["X"]
    id_pliku = dane["id_pliku"]

    print(f"Wczytano {len(X)} okien.")
    print(f"Horyzont predykcji H = {HORYZONT_H} okien")

    odchylki = biezace_odchylki(X, C_WZORZEC)
    print(f"Biezace odchylki WZGLEDNE (ulamek): min {odchylki.min():.4f}, "
          f"mediana {np.median(odchylki):.4f}, max {odchylki.max():.4f}")

    y = etykiety_predykcyjne(odchylki, id_pliku, HORYZONT_H, PROG_WZGLEDNY)

    maska = y >= 0           # okna z wazna etykieta (nie konce plikow)
    n_ok = int(maska.sum())
    n_popr = int((y[maska] == 0).sum())
    n_odch = int((y[maska] == 1).sum())

    print(f"\nProg wzgledny: {PROG_WZGLEDNY} (ulamek odstepstwa)")
    print(f"Okien z etykieta: {n_ok}  (pominieto {len(X)-n_ok} koncow plikow)")
    print(f"  poprawne (0): {n_popr}")
    print(f"  odchylki (1): {n_odch}  ({100*n_odch/n_ok:.1f}%)")

    # zapisujemy TYLKO okna z wazna etykieta (maska), zeby 03_siec.py dostal czyste dane
    np.savez(
        PLIK_WYJSCIOWY,
        X=X[maska],
        y=y[maska],
        id_pliku=id_pliku[maska],
        odchylki=odchylki[maska],
    )
    print(f"\nZapisano do: {PLIK_WYJSCIOWY}  (horyzont H={HORYZONT_H})")


if __name__ == "__main__":
    main()
