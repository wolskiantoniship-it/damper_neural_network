"""
04_horyzonty.py   (wersja z powtarzaniem - srednia +- odchylenie)
==================================================================
Co robi (po ludzku):
Sprawdza, jak daleko w przod da sie przewidziec odchylke - ale teraz
KAZDY horyzont liczymy WIELE RAZY z roznym losowym podzialem plikow.
Dzieki temu zamiast jednej liczby AUROC dostajemy SREDNIA +- ODCHYLENIE.

Po co? Bo przy 18 plikach test opiera sie na zaledwie ~3 plikach. Jedna
liczba (np. 0.97) moze byc szczesciem przy konkretnym podziale. Powtarzajac
z roznym ziarnem widzimy, czy wynik jest STABILNY, czy skacze.

Dla kazdego H i kazdego powtorzenia liczymy te same dwie rzeczy:
  - AUROC sieci (prawdziwa predykcja),
  - AUROC "sciagi" (czy biezaca odchylka sama zdradza przyszlosc).

Korzysta z tej samej logiki etykiet i stalych (z parametry.py) oraz
tej samej architektury sieci co 05_siec.py.

Wejscie:  wyniki/okna_dane.npz   (z 01_parsowanie.py)
Wyjscie:  wyniki/horyzonty.png, wyniki/04_horyzonty_log.txt

Uruchomienie:
    python 04_horyzonty.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from parametry import (
    C_WZORZEC, PROG_WZGLEDNY,
    biezace_odchylki, etykiety_predykcyjne,
    sciezka_wyniku, Logger,
)

# --- ustawienia ---
PLIK_DANE = "okna_dane.npz"
HORYZONTY = [2, 5, 10, 20, 40]
EPOKI = 25
UDZIAL_TEST = 0.2
UDZIAL_WAL = 0.2

# ile razy powtorzyc kazdy horyzont (rozne ziarna = rozne podzialy plikow)
LICZBA_POWTORZEN = 7


log = Logger("04_horyzonty_log.txt")


def podziel_po_plikach(id_pliku, ziarno):
    """Podzial po plikach - z konkretnym ziarnem (rozne ziarno = inny podzial)."""
    pliki = np.unique(id_pliku)
    rng = np.random.default_rng(ziarno)
    rng.shuffle(pliki)
    n = len(pliki)
    n_test = max(1, int(n * UDZIAL_TEST))
    n_wal = max(1, int(n * UDZIAL_WAL))
    test = np.isin(id_pliku, pliki[:n_test])
    wal = np.isin(id_pliku, pliki[n_test:n_test + n_wal])
    tren = np.isin(id_pliku, pliki[n_test + n_wal:])
    return tren, wal, test


def zbuduj_siec(ksztalt):
    model = keras.Sequential([
        keras.Input(shape=ksztalt),
        layers.Conv1D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def jeden_przebieg(X, yv, idp, ziarno):
    """Jeden trening sieci z danym ziarnem. Zwraca AUROC na tescie albo None."""
    tf.random.set_seed(ziarno)
    tren, wal, test = podziel_po_plikach(idp, ziarno)

    if (yv[test] == 1).sum() < 1 or (yv[test] == 0).sum() < 1:
        return None   # test bez obu klas - nie da sie policzyc AUROC
    if (yv[tren] == 1).sum() < 1 or (yv[tren] == 0).sum() < 1:
        return None

    sr = X[tren].mean((0, 1), keepdims=True)
    od = X[tren].std((0, 1), keepdims=True) + 1e-8
    Xtr, Xwl, Xte = (X[tren]-sr)/od, (X[wal]-sr)/od, (X[test]-sr)/od

    n0, n1 = (yv[tren] == 0).sum(), (yv[tren] == 1).sum()
    waga = {0: 1.0, 1: float(n0)/max(1, n1)}

    model = zbuduj_siec(Xtr.shape[1:])
    model.fit(Xtr, yv[tren], validation_data=(Xwl, yv[wal]),
              epochs=EPOKI, batch_size=32, class_weight=waga, verbose=0)
    prob = model.predict(Xte, verbose=0).ravel()
    return roc_auc_score(yv[test], prob)


def main():
    dane = np.load(sciezka_wyniku(PLIK_DANE), allow_pickle=True)
    X_all = dane["X"].astype("float32")
    id_all = dane["id_pliku"]
    odch_all = biezace_odchylki(X_all, C_WZORZEC)

    H_uzyte = []
    siec_srednia, siec_odchyl = [], []
    sciaga_srednia = []

    for H in HORYZONTY:
        y = etykiety_predykcyjne(odch_all, id_all, H, PROG_WZGLEDNY)
        maska = y >= 0
        X, yv, idp, odch = X_all[maska], y[maska], id_all[maska], odch_all[maska]

        if (yv == 1).sum() < 5 or (yv == 0).sum() < 5:
            log(f"H={H}: za malo przykladow jednej klasy - pomijam")
            continue

        # sciaga nie zalezy od ziarna (to staly pomiar) - liczymy raz
        auc_sciaga = roc_auc_score(yv, odch)

        # siec: powtarzamy z roznymi ziarnami
        wyniki = []
        for p in range(LICZBA_POWTORZEN):
            a = jeden_przebieg(X, yv, idp, ziarno=100 + p)
            if a is not None:
                wyniki.append(a)

        if not wyniki:
            log(f"H={H}: zaden podzial nie dal obu klas w tescie - pomijam")
            continue

        wyniki = np.array(wyniki)
        H_uzyte.append(H)
        siec_srednia.append(wyniki.mean())
        siec_odchyl.append(wyniki.std())
        sciaga_srednia.append(auc_sciaga)

        log(f"H={H:3d}: siec {wyniki.mean():.3f} +- {wyniki.std():.3f} "
            f"(z {len(wyniki)} powt.) | sciaga {auc_sciaga:.3f}")

    # --- wykres ze slupkami bledu ---
    H_uzyte = np.array(H_uzyte)
    siec_srednia = np.array(siec_srednia)
    siec_odchyl = np.array(siec_odchyl)

    plt.figure(figsize=(8, 5))
    plt.errorbar(H_uzyte, siec_srednia, yerr=siec_odchyl, fmt="o-",
                 capsize=4, label="siec (srednia +- odch.)")
    plt.plot(H_uzyte, sciaga_srednia, "s--", label="sciaga (biezaca odchylka)")
    plt.axhline(0.5, color="gray", linestyle=":", label="losowy (0.5)")
    plt.xlabel("horyzont predykcji H [okien]")
    plt.ylabel("AUROC")
    plt.title("Predykcja odchylki - srednia z powtorzen +- odchylenie")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0.3, 1.05)
    plt.tight_layout()
    sciezka_png = sciezka_wyniku("horyzonty.png")
    plt.savefig(sciezka_png, dpi=120)
    plt.close()
    log(f"\nWykres zapisany do: {sciezka_png}")
    log(f"(kazdy punkt sieci = srednia z {LICZBA_POWTORZEN} podzialow, slupek = odchylenie)")
    log("\nJAK CZYTAC: maly slupek bledu = wynik STABILNY (nie zalezy od podzialu).")
    log("Duzy slupek = wynik niepewny, zalezny od tego ktore pliki w tescie.")
    log.zapisz()


if __name__ == "__main__":
    main()
