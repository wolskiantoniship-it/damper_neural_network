"""
05_siec.py
==========
Co robi ten skrypt (po ludzku):
1. Wczytuje okna + etykiety (wynik z 02_etykietuj.py).
2. Dzieli dane na trening / walidacje / test - ale UWAGA: dzieli PO PLIKACH,
   nie po oknach. Czemu? Bo okna z jednego pliku sa do siebie podobne. Gdyby
   czesc trafila do treningu a czesc do testu, siec "oszukiwalaby" znajac
   odpowiedzi. Dzielac po plikach test jest uczciwy.
3. Skaluje dane (z-score) - liczone TYLKO z treningu, zeby test byl czysty.
4. Trenuje mala siec konwolucyjna 1D (CNN1D) w Keras.
5. Liczy i rysuje: krzywa ROC + AUROC, precision, recall, accuracy, F1,
   krzywa precision-recall, macierz pomylek.

Wejscie:  wyniki/okna_etykiety.npz   (z 02_etykietuj.py)
Wyjscie:  wyniki/wyniki.png, wyniki/model_amortyzator.keras,
          wyniki/model_parametry.npz, wyniki/05_siec_log.txt

Uruchomienie:
    python 05_siec.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, accuracy_score, f1_score,
    confusion_matrix, average_precision_score,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from parametry import sciezka_wyniku, Logger

# powtarzalnosc wynikow
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# USTAWIENIA
# ============================================================

PLIK_WEJSCIOWY = "okna_etykiety.npz"
EPOKI = 40
ROZMIAR_BATCHA = 32

# Jak dzielimy pliki: jaka czesc na test i walidacje (reszta na trening).
UDZIAL_TEST = 0.2
UDZIAL_WALIDACJA = 0.2


log = Logger("05_siec_log.txt")


# ============================================================
# PRZYGOTOWANIE DANYCH
# ============================================================

def podziel_po_plikach(id_pliku):
    """
    Dzieli ZBIOR PLIKOW (nie okien) na trening/walidacje/test.
    Zwraca maski (True/False) dla okien nalezacych do kazdej czesci.
    """
    pliki = np.unique(id_pliku)
    rng = np.random.default_rng(SEED)
    rng.shuffle(pliki)

    n = len(pliki)
    n_test = max(1, int(n * UDZIAL_TEST))
    n_wal = max(1, int(n * UDZIAL_WALIDACJA))

    pliki_test = pliki[:n_test]
    pliki_wal = pliki[n_test:n_test + n_wal]
    pliki_tren = pliki[n_test + n_wal:]

    maska_test = np.isin(id_pliku, pliki_test)
    maska_wal = np.isin(id_pliku, pliki_wal)
    maska_tren = np.isin(id_pliku, pliki_tren)

    log(f"Pliki -> trening: {len(pliki_tren)}, walidacja: {len(pliki_wal)}, test: {len(pliki_test)}")
    return maska_tren, maska_wal, maska_test


def skaluj(X_tren, X_wal, X_test):
    """
    Z-score: (wartosc - srednia) / odchylenie.
    Srednia i odchylenie liczymy TYLKO z treningu (zeby nie podgladac testu).
    Liczymy osobno dla kazdej z 3 kolumn (sila, przem, predkosc).
    """
    srednia = X_tren.mean(axis=(0, 1), keepdims=True)
    odchyl = X_tren.std(axis=(0, 1), keepdims=True) + 1e-8

    return (
        (X_tren - srednia) / odchyl,
        (X_wal - srednia) / odchyl,
        (X_test - srednia) / odchyl,
    )


# ============================================================
# SIEC
# ============================================================

def zbuduj_siec(ksztalt_wejscia):
    """
    Mala siec konwolucyjna 1D.
    Conv1D "przesuwa sie" po oknie czasowym i wylapuje wzorce w przebiegu.
    Na koncu jeden neuron z sigmoidem -> prawdopodobienstwo etykiety 1.
    """
    model = keras.Sequential([
        keras.Input(shape=ksztalt_wejscia),
        layers.Conv1D(16, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# WYKRESY
# ============================================================

def zrob_wykresy(y_test, y_prob, y_pred):
    """Rysuje wszystkie wymagane wykresy do jednego pliku PNG."""
    fig, osie = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Krzywa ROC + AUROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auroc = roc_auc_score(y_test, y_prob)
    ax = osie[0, 0]
    ax.plot(fpr, tpr, label=f"ROC (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="losowy")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Krzywa ROC")
    ax.legend()

    # 2) Krzywa Precision-Recall
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax = osie[0, 1]
    ax.plot(rec, prec, label=f"PR (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Krzywa Precision-Recall")
    ax.legend()

    # 3) Macierz pomylek
    cm = confusion_matrix(y_test, y_pred)
    ax = osie[1, 0]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["poprawne (0)", "odchylka (1)"])
    ax.set_yticklabels(["poprawne (0)", "odchylka (1)"])
    ax.set_xlabel("Przewidziane")
    ax.set_ylabel("Prawdziwe")
    ax.set_title("Macierz pomylek")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black", fontsize=14)

    # 4) Slupki z metrykami
    metryki = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    ax = osie[1, 1]
    ax.bar(metryki.keys(), metryki.values())
    ax.set_ylim(0, 1.12)
    ax.set_title("Metryki na zbiorze testowym", pad=12)
    for i, (k, v) in enumerate(metryki.items()):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

    plt.tight_layout()
    sciezka_png = sciezka_wyniku("wyniki.png")
    plt.savefig(sciezka_png, dpi=120)
    plt.close(fig)
    log(f"\nWykresy zapisane do: {sciezka_png}")
    return metryki, auroc


# ============================================================
# GLOWNA CZESC
# ============================================================

def main():
    dane = np.load(sciezka_wyniku(PLIK_WEJSCIOWY), allow_pickle=True)
    X = dane["X"].astype("float32")
    y = dane["y"].astype("int")
    id_pliku = dane["id_pliku"]

    log(f"Dane: {X.shape[0]} okien, ksztalt okna {X.shape[1:]}")
    log(f"Rozklad etykiet -> poprawne: {(y==0).sum()}, odchylki: {(y==1).sum()}")

    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        log("\nUWAGA: wszystkie okna maja te sama etykiete!")
        log("Sieci nie ma sensu trenowac. Wroc do 02_etykietuj.py i dobierz PROG,")
        log("albo dodaj wiecej plikow z roznymi amortyzatorami.")
        log.zapisz()
        return

    maska_tren, maska_wal, maska_test = podziel_po_plikach(id_pliku)

    X_tren, X_wal, X_test = skaluj(X[maska_tren], X[maska_wal], X[maska_test])
    y_tren, y_wal, y_test = y[maska_tren], y[maska_wal], y[maska_test]

    # zapamietujemy parametry skalowania (z treningu) i ktore pliki sa testowe -
    # 06_zastosowanie.py uzyje ich, by odtworzyc DOKLADNIE ten sam stan bez treningu
    srednia_skali = X[maska_tren].mean(axis=(0, 1), keepdims=True)
    odchyl_skali = X[maska_tren].std(axis=(0, 1), keepdims=True) + 1e-8
    pliki_testowe = np.unique(id_pliku[maska_test])

    # waga klas - jak odchylek jest malo, mowimy sieci ze sa wazniejsze
    n0, n1 = (y_tren == 0).sum(), (y_tren == 1).sum()
    waga_klas = {0: 1.0, 1: float(n0) / max(1, n1)}

    model = zbuduj_siec(X_tren.shape[1:])
    model.summary(print_fn=log)

    model.fit(
        X_tren, y_tren,
        validation_data=(X_wal, y_wal),
        epochs=EPOKI,
        batch_size=ROZMIAR_BATCHA,
        class_weight=waga_klas,
        verbose=2,
    )

    # przewidywania na tescie
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metryki, auroc = zrob_wykresy(y_test, y_prob, y_pred)

    log("\n=== WYNIKI NA TESCIE ===")
    log(f"AUROC:     {auroc:.3f}")
    for nazwa, wartosc in metryki.items():
        log(f"{nazwa:10s} {wartosc:.3f}")

    sciezka_model = sciezka_wyniku("model_amortyzator.keras")
    model.save(sciezka_model)
    log(f"\nModel zapisany do: {sciezka_model}")

    # zapisujemy parametry potrzebne 06_zastosowanie.py, by uzyc tej samej sieci
    sciezka_param = sciezka_wyniku("model_parametry.npz")
    np.savez(
        sciezka_param,
        srednia=srednia_skali,
        odchyl=odchyl_skali,
        pliki_testowe=pliki_testowe,
    )
    log(f"Parametry (skalowanie + pliki testowe) zapisane do: {sciezka_param}")
    log.zapisz()


if __name__ == "__main__":
    main()
