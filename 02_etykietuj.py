"""
02_etykietuj.py   (wersja PREDYKCYJNA)
==================================================
Co robi (po ludzku):
Nadaje kazdemu oknu etykiete 0/1, ale etykieta dotyczy PRZYSZLOSCI:
  etykieta okna n = 1, jesli za H okien (czyli w oknie n+H) sila odjedzie
  od normy bardziej niz prog.
Sieci pokazujemy okno n (terazniejszosc), a ona ma przewidziec, czy nadchodzi
odchylka. To jest "wczesne ostrzeganie" - przewidywanie narastajacego trendu.

DLACZEGO to nie jest trywialne:
Etykieta opisuje okno n+H, ktorego siec NIE widzi. Zeby trafic, musi wylapac
w biezacym oknie zapowiedz trendu, a nie po prostu odczytac biezaca wartosc.

NORMA (wzorzec):
Sila oczekiwana = C * predkosc (krzywa wzorcowa z modelu liniowego).
Biezaca odchylka okna = jak bardzo zmierzona sila odbiega od C*v (wzglednie).

WAZNE: okna n i n+H musza byc z TEGO SAMEGO pliku. Okna przy koncu pliku,
dla ktorych nie ma n+H, dostaja etykiete -1 i sa pomijane.

Stale i funkcje (PROG_WZGLEDNY, HORYZONT_H, C_WZORZEC, EPS_KN,
biezace_odchylki, etykiety_predykcyjne) sa w pliku parametry.py - tam je zmieniasz.

Wejscie:  wyniki/okna_dane.npz   (z 01_parsowanie.py)
Wyjscie:  wyniki/okna_etykiety.npz

Uruchomienie:
    python 02_etykietuj.py
"""

import numpy as np
from parametry import (
    C_WZORZEC, HORYZONT_H, PROG_WZGLEDNY,
    biezace_odchylki, etykiety_predykcyjne,
    sciezka_wyniku, Logger,
)

PLIK_WEJSCIOWY = "okna_dane.npz"
PLIK_WYJSCIOWY = "okna_etykiety.npz"


def main():
    log = Logger("02_etykietuj_log.txt")

    dane = np.load(sciezka_wyniku(PLIK_WEJSCIOWY), allow_pickle=True)
    X = dane["X"]
    id_pliku = dane["id_pliku"]

    log(f"Wczytano {len(X)} okien.")
    log(f"Horyzont predykcji H = {HORYZONT_H} okien")

    odchylki = biezace_odchylki(X, C_WZORZEC)
    log(f"Biezace odchylki WZGLEDNE (ulamek): min {odchylki.min():.4f}, "
        f"mediana {np.median(odchylki):.4f}, max {odchylki.max():.4f}")

    y = etykiety_predykcyjne(odchylki, id_pliku, HORYZONT_H, PROG_WZGLEDNY)

    maska = y >= 0           # okna z wazna etykieta (nie konce plikow)
    n_ok = int(maska.sum())
    n_popr = int((y[maska] == 0).sum())
    n_odch = int((y[maska] == 1).sum())

    log(f"\nProg wzgledny: {PROG_WZGLEDNY} (skala ulamka odstepstwa od wzorca)")
    log(f"Okien z etykieta: {n_ok}  (pominieto {len(X)-n_ok} koncow plikow)")
    log(f"  poprawne (0): {n_popr}")
    log(f"  odchylki (1): {n_odch}  ({100*n_odch/n_ok:.1f}%)")

    # zapisujemy TYLKO okna z wazna etykieta (maska), zeby 05_siec.py dostal czyste dane
    sciezka_npz = sciezka_wyniku(PLIK_WYJSCIOWY)
    np.savez(
        sciezka_npz,
        X=X[maska],
        y=y[maska],
        id_pliku=id_pliku[maska],
        odchylki=odchylki[maska],
    )
    log(f"\nZapisano do: {sciezka_npz}  (horyzont H={HORYZONT_H})")
    log.zapisz()


if __name__ == "__main__":
    main()
