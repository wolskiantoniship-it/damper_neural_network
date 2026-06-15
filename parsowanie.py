import numpy as np 
from pathlib import Path



FOLDER_Z_DANYMI = r"C:\Users\wolsk\Desktop\SPA2026_2\dane"
LINIE_NAGLOWKA = 8
DLUGOSC_OKNA = 20 #dlugosc jednej probki
ZAKLADKA = 0.5 # probki nachodza na siebie w 50 procentach
PLIK_WYJSCIOWY = "okna_dane.npz" # Gdzie zapisac wynik.


# ============================================================
def polska_liczba(tekst):
    """Zamienia '-0,2281' (polski zapis) na liczbe -0.2281."""
    return float(tekst.strip().strip('"').replace(",", "."))


def wczytaj_jeden_plik(sciezka): #Wczytuje jeden plik CSV. Zwraca trzy tablice: sila [kN], przemieszczenie [mm], czas [s].
    sila, przemieszczenie, czas = [], [], []
    with open(sciezka, encoding="utf-8-sig") as f:
        linie = f.readlines()
    # pomijam naglowek i lece po wierszach z danymi
    for linia in linie[LINIE_NAGLOWKA:]:
        linia = linia.strip()
        if not linia:
            continue
        czesci = linia.split(";")
        if len(czesci) < 3:
            continue  # pomijamy popsute/niepelne wiersze
        try:
            sila.append(polska_liczba(czesci[0]))
            przemieszczenie.append(polska_liczba(czesci[1]))
            czas.append(polska_liczba(czesci[2]))
        except ValueError:
            continue  # jak sie nie da sparsowac, pomijamy wiersz
    return np.array(sila), np.array(przemieszczenie), np.array(czas)


def policz_predkosc(przemieszczenie, czas): #Liczy predkosc = zmiana przemieszczenia / zmiana czasu.
    return np.gradient(przemieszczenie, czas) #np.gradient daje predkosc dla kazdego punktu (radzi sobie z brzegami).


def potnij_na_okna(sila, przemieszczenie, predkosc, dlugosc, zakladka):
    """ Tnie dlugie przebiegi na krotkie okna.
    Kazde okno to kawalek sygnalu o ksztalcie (dlugosc, 3),
    gdzie 3 kolumny to: sila, przemieszczenie, predkosc. """
    krok = int(dlugosc * (1 - zakladka))
    if krok < 1:
        krok = 1

    okna = []
    n = len(sila)
    start = 0
    while start + dlugosc <= n:
        okno = np.column_stack([
            sila[start:start + dlugosc],
            przemieszczenie[start:start + dlugosc],
            predkosc[start:start + dlugosc],
        ])
        okna.append(okno)
        start += krok

    return okna



def main():
    folder = Path(FOLDER_Z_DANYMI)
    pliki = sorted(folder.glob("*.csv")) 

    if not pliki:
        print(f"Nie znalazlem zadnych plikow CSV w: {folder.resolve()}")
        return

    print(f"Znalazlem {len(pliki)} plikow CSV.\n")

    wszystkie_okna = []      # tu zbieramy wszystkie okna ze wszystkich plikow
    id_pliku_dla_okna = []   # dla kazdego okna - z ktorego pliku pochodzi
    nazwy_plikow = []        # lista nazw plikow (zeby wiedziec co jest czym)

    for nr_pliku, sciezka in enumerate(pliki):
        sila, przem, czas = wczytaj_jeden_plik(sciezka)

        if len(sila) < DLUGOSC_OKNA:
            print(f"  [POMIJAM] {sciezka.name}: za malo danych ({len(sila)} probek)")
            continue

        predkosc = policz_predkosc(przem, czas)
        okna = potnij_na_okna(sila, przem, predkosc, DLUGOSC_OKNA, ZAKLADKA)

        wszystkie_okna.extend(okna)
        id_pliku_dla_okna.extend([nr_pliku] * len(okna))
        nazwy_plikow.append(sciezka.name)

        print(f"  {sciezka.name}: {len(sila)} probek -> {len(okna)} okien")

    # zamiana list na tablice numpy
    X = np.array(wszystkie_okna)              # ksztalt: (liczba_okien, dlugosc, 3)
    id_pliku = np.array(id_pliku_dla_okna)    # ksztalt: (liczba_okien,)

    print(f"\nRAZEM: {X.shape[0]} okien, kazde o ksztalcie {X.shape[1:]} (probki x [sila, przem, predkosc])")

    np.savez(
        PLIK_WYJSCIOWY,
        X=X,
        id_pliku=id_pliku,
        nazwy_plikow=np.array(nazwy_plikow),
    )
    print(f"Zapisano do: {PLIK_WYJSCIOWY}")


if __name__ == "__main__":
    main()
