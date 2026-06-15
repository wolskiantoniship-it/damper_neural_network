"""
Nadaje kazdemu oknu etykiete 0 (poprawne) albo 1 (odchylka od normy).

POMYSL JEST TAKI:
Amortyzator opisuje krzywa SILA-PREDKOSC (tzw. charakterystyka tlumienia).
Mamy "wzorzec" - jak ta krzywa powinna wygladac dla zdrowego amora.
Dla kazdego okna sprawdzamy: jak bardzo zmierzona sila odbiega od wzorca
przy tej samej predkosci? Jesli odbiega bardziej niz PROG -> etykieta 1.

"""
#dobrze przejrzeć dane trenujące, policzyc je i zbalansować - mniej więcej porówno
import numpy as np
# ============================================================
PLIK_WEJSCIOWY = "okna_dane.npz"    
PLIK_WYJSCIOWY = "okna_etykiety.npz" 
TRYB = "simulink"
PLIK_WZORCA_SIMULINK = "wzorzec_simulink.csv"
# === WAŻNY PARAMETR ===
PROG_ODCHYLKI_KN = 0.15
# "srednia" = przecietne odchylenie w oknie (lagodniejsze).
# "max"     = najwiekszy wyskok w oknie (czulsze na pojedyncze skoki).
SPOSOB_LICZENIA = "srednia"


# ============================================================
def wczytaj_wzorzec_z_danych(X):
    """ 
    tryb dane - pozostałość po testach liczył medianę z naszych danych i względem niej definiował odchyłkę
    """
    # rozpłaszczamy wszystkie okna do listy punktow
    predkosc = X[:, :, 2].ravel()
    sila = X[:, :, 0].ravel()
    # dzielimy zakres predkosci na koszyki i w kazdym liczymy mediane sily
    liczba_koszykow = 40
    granice = np.linspace(predkosc.min(), predkosc.max(), liczba_koszykow + 1)
    srodki = 0.5 * (granice[:-1] + granice[1:])
    mediany = np.full(liczba_koszykow, np.nan)
    for i in range(liczba_koszykow):
        maska = (predkosc >= granice[i]) & (predkosc < granice[i + 1])
        if np.any(maska):
            mediany[i] = np.median(sila[maska])
    # uzupelniamy ewentualne dziury (koszyki bez punktow) interpolacja
    dobre = ~np.isnan(mediany)
    mediany = np.interp(srodki, srodki[dobre], mediany[dobre])

    def wzorzec(v):
        return np.interp(v, srodki, mediany)

    return wzorzec


def wczytaj_wzorzec_z_simulink(sciezka):
    """
    Tryb "simulink": wczytujemy krzywa wzorcowa z pliku.
    Format: dwie kolumny - predkosc[mm/s], sila[kN].
    """
    dane = np.loadtxt(sciezka, delimiter=",")
    v_wzor = dane[:, 0]
    f_wzor = dane[:, 1]
    kolejnosc = np.argsort(v_wzor)
    v_wzor, f_wzor = v_wzor[kolejnosc], f_wzor[kolejnosc]

    def wzorzec(v):
        return np.interp(v, v_wzor, f_wzor)

    return wzorzec


def policz_etykiety(X, wzorzec):
    """ Dla kazdego okna liczy odchylke zmierzonej sily od wzorca
    i porownuje z progiem.
    """
    etykiety = np.zeros(len(X), dtype=int)
    odchylki = np.zeros(len(X))

    for i, okno in enumerate(X):
        sila = okno[:, 0]
        predkosc = okno[:, 2]
        sila_wzorcowa = wzorzec(predkosc)
        roznica = np.abs(sila - sila_wzorcowa)

        if SPOSOB_LICZENIA == "max":
            odchylka = roznica.max()
        else:
            odchylka = roznica.mean()
        odchylki[i] = odchylka
        etykiety[i] = 1 if odchylka > PROG_ODCHYLKI_KN else 0
    return etykiety, odchylki


# ============================================================
def main():
    dane = np.load(PLIK_WEJSCIOWY, allow_pickle=True)
    X = dane["X"]
    id_pliku = dane["id_pliku"]

    print(f"Wczytano {len(X)} okien.")
    print(f"Tryb etykietowania: {TRYB}")

    if TRYB == "simulink":
        wzorzec = wczytaj_wzorzec_z_simulink(PLIK_WZORCA_SIMULINK)
    else:
        wzorzec = wczytaj_wzorzec_z_danych(X)

    etykiety, odchylki = policz_etykiety(X, wzorzec)

    ile_zlych = int(etykiety.sum())
    print(f"\nProg odchylki: {PROG_ODCHYLKI_KN} kN (sposob: {SPOSOB_LICZENIA})")
    print(f"Okna poprawne (0): {len(etykiety) - ile_zlych}")
    print(f"Okna z odchylka (1): {ile_zlych}  ({100*ile_zlych/len(etykiety):.1f}%)")
    print(f"\nOdchylki - min: {odchylki.min():.3f}, mediana: {np.median(odchylki):.3f}, max: {odchylki.max():.3f} kN")
    print("(Te liczby pomagaja Ci dobrac PROG - patrz gdzie lezy granica.)")

    np.savez(
        PLIK_WYJSCIOWY,
        X=X,
        y=etykiety,
        id_pliku=id_pliku,
        odchylki=odchylki,
    )
    print(f"\nZapisano do: {PLIK_WYJSCIOWY}")
if __name__ == "__main__":
    main()
