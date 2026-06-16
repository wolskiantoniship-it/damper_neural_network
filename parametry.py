"""
parametry.py
============
Jedno miejsce na WSPOLNE stale, sciezki i funkcje uzywane przez caly pipeline.

Po co to istnieje:
Funkcje liczace odchylke i etykiety oraz stale (prog, horyzont, wzorzec)
byly wczesniej KOPIOWANE w kilku plikach. To grozilo desynchronizacja -
wystarczylo zmienic prog w jednym miejscu i zapomniec w drugim, a wyniki
przestawaly do siebie pasowac. Teraz wszystko jest TUTAJ, a pozostale
skrypty robia `from parametry import ...`. Jedno zrodlo prawdy.

Jesli chcesz zmienic prog / horyzont / wzorzec - zmieniasz TYLKO tutaj.

Zawiera tez:
  - FOLDER_WYNIKI: wszystkie wyniki (PNG, NPZ, model, logi) laduja tutaj,
  - sciezka_wyniku(): buduje pelna sciezke do pliku w folderze wynikow,
  - Logger: prosty pomocnik do zapisywania logow tekstowych (.txt).
"""

import os
import numpy as np


# ============================================================
# STALE WSPOLNE DLA CALEGO PIPELINE
# ============================================================

# wspolczynnik krzywej wzorcowej z modelu liniowego F = C * v
C_WZORZEC = 0.00278

# horyzont predykcji: ile okien w przod patrzy etykieta
HORYZONT_H = 20

# prog odchylki WZGLEDNEJ (bezwymiarowy, w skali ulamka odstepstwa od wzorca).
# UWAGA: to NIE jest 0.04 - przy obecnych danych mediana odchylek ~2.3,
# a max ~9.1, wiec prog 4 oznacza "okno odbiega srednio ~4x bardziej
# niz typowo wynosi odstepstwo od wzorca". Dobrany z podgladu progu (03).
PROG_WZGLEDNY = 3.5

# epsilon chroni przed dzieleniem przez ~0, gdy tlok zawraca (v~0 -> C*v~0).
# Jednostka: kN. Dobrany jako maly ulamek typowej sily.
EPS_KN = 0.05


# ============================================================
# SCIEZKI / FOLDER WYNIKOW
# ============================================================

# Wszystkie wyniki (wykresy PNG, pliki NPZ, model, logi TXT) laduja tutaj.
FOLDER_WYNIKI = "wyniki"


def sciezka_wyniku(nazwa):
    """
    Buduje pelna sciezke do pliku w folderze wynikow i zapewnia,
    ze folder istnieje. Uzywaj zawsze, gdy cos zapisujesz.
    """
    os.makedirs(FOLDER_WYNIKI, exist_ok=True)
    return os.path.join(FOLDER_WYNIKI, nazwa)


class Logger:
    """
    Prosty pomocnik: wypisuje na ekran I zapamietuje, a na koniec
    zapisuje caly log do pliku .txt w folderze wynikow.

    Uzycie:
        log = Logger("01_parsowanie_log.txt")
        log("cokolwiek")
        ...
        log.zapisz()
    """

    def __init__(self, nazwa_pliku):
        self.nazwa_pliku = nazwa_pliku
        self._linie = []

    def __call__(self, tekst=""):
        print(tekst)
        self._linie.append(str(tekst))

    def zapisz(self):
        sciezka = sciezka_wyniku(self.nazwa_pliku)
        with open(sciezka, "w", encoding="utf-8") as f:
            f.write("\n".join(self._linie))
        print(f"\nLog zapisany do: {sciezka}")


# ============================================================
# WSPOLNE FUNKCJE
# ============================================================

def biezace_odchylki(X, C=C_WZORZEC):
    """
    Dla kazdego okna: srednie WZGLEDNE odstepstwo sily od wzorca.
    = |zmierzona - C*v| / (|C*v| + eps)
    Dzielenie przez wartosc wzorcowa usuwa zaleznosc od bezwzglednej
    predkosci: ta sama % odchylka znaczy to samo przy wolnym i szybkim
    ruchu tloka. eps zapobiega wybuchowi przy zawracaniu tloka (v~0).
    """
    wynik = []
    for o in X:
        F = o[:, 0]
        v = o[:, 2]
        wzorzec = C * v
        wzgledne = np.abs(F - wzorzec) / (np.abs(wzorzec) + EPS_KN)
        wynik.append(wzgledne.mean())
    return np.array(wynik)


def etykiety_predykcyjne(odchylki, id_pliku, H=HORYZONT_H, prog=PROG_WZGLEDNY):
    """
    Etykieta okna n = 1, jesli odchylka okna (n+H) > prog.
    Tylko gdy okna n i n+H sa z TEGO SAMEGO pliku; inaczej -1 (pomijamy).

    Dzieki temu siec widzi okno n (terazniejszosc), a uczy sie przewidziec,
    czy za H okien NADEJDZIE odchylka - to jest "wczesne ostrzeganie".
    """
    n = len(odchylki)
    y = np.full(n, -1, dtype=int)
    for i in range(n):
        j = i + H
        if j < n and id_pliku[j] == id_pliku[i]:
            y[i] = 1 if odchylki[j] > prog else 0
    return y
