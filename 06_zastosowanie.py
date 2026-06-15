"""
06_zastosowanie.py
==================
Co robi (po ludzku):
Pokazuje, jak wytrenowana siec dzialalaby W PRAKTYCE jako system wczesnego
ostrzegania. UZYWA GOTOWEJ sieci z 05_siec.py (nie trenuje od nowa!), wiec
demonstruje DOKLADNIE ten model, ktorego metryki podajesz w artykule.

Najpierw uruchom 05_siec.py - on zapisuje (do folderu wynikow):
  - model_amortyzator.keras  (wytrenowana siec)
  - model_parametry.npz       (skalowanie + ktore pliki sa testowe)
06_zastosowanie.py wczytuje oba i przejezdza pliki testowe okno po oknie.

Liczy TRZY rzeczy:
  WIDOK 1 - symulacja na zywo (wykres alarmow w czasie),
  WIDOK 2 - werdykt per plik (zdrowy / wymaga uwagi),
  WIDOK 3 - ocena per prog decyzyjny (alarmy / trafione / falszywe).

Stale (HORYZONT_H, PROG_WZGLEDNY, C_WZORZEC, EPS_KN) pochodza z parametry.py,
wiec sa automatycznie spojne z 02_etykietuj.py.

Wejscie:  wyniki/okna_dane.npz, wyniki/model_amortyzator.keras,
          wyniki/model_parametry.npz

Uruchomienie:
    python 06_zastosowanie.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from parametry import (
    C_WZORZEC, HORYZONT_H, PROG_WZGLEDNY,
    biezace_odchylki, etykiety_predykcyjne,
    sciezka_wyniku, Logger,
)

# === PLIKI (w folderze wynikow) ===
PLIK_DANE = "okna_dane.npz"
PLIK_MODEL = "model_amortyzator.keras"
PLIK_PARAM = "model_parametry.npz"

# progi decyzyjne sieci do porownania (osobne od progu etykiet!)
PROGI_ALARMU = [0.3, 0.5, 0.7]

# czestotliwosc probkowania i dlugosc okna - do przeliczenia okien na SEKUNDY
HZ = 10.0                # probkowanie DAQ [Hz]
DLUGOSC_OKNA = 20        # probek na okno (z 01_parsowanie.py)
ZAKLADKA = 0.5           # zakladka okien (z 01_parsowanie.py)


log = Logger("06_zastosowanie_log.txt")


def krok_czasu_okna():
    """Ile SEKUND dzieli poczatki dwoch sasiednich okien."""
    krok_probek = DLUGOSC_OKNA * (1 - ZAKLADKA)
    return krok_probek / HZ


def main():
    dane = np.load(sciezka_wyniku(PLIK_DANE), allow_pickle=True)
    X = dane["X"].astype("float32")
    idp = dane["id_pliku"]

    odch = biezace_odchylki(X, C_WZORZEC)
    y = etykiety_predykcyjne(odch, idp, HORYZONT_H, PROG_WZGLEDNY)
    maska = y >= 0
    X, y, idp, odch = X[maska], y[maska], idp[maska], odch[maska]

    # --- WCZYTUJEMY gotowa siec i parametry (zamiast trenowac) ---
    try:
        model = keras.models.load_model(sciezka_wyniku(PLIK_MODEL))
        param = np.load(sciezka_wyniku(PLIK_PARAM), allow_pickle=True)
    except (OSError, IOError, ValueError) as e:
        log("Nie moge wczytac modelu/parametrow.")
        log(f"Blad: {e}")
        log("\nNajpierw uruchom 05_siec.py - on tworzy (w folderze wynikow):")
        log("  model_amortyzator.keras  oraz  model_parametry.npz")
        log.zapisz()
        return

    srednia = param["srednia"]
    odchyl = param["odchyl"]
    pliki_test = param["pliki_testowe"]
    log(f"Wczytano gotowa siec. Pliki testowe (z 05_siec.py): {sorted(pliki_test)}")

    # bierzemy DOKLADNIE te pliki, ktore byly testowe w 05 (siec ich nie widziala)
    m_test = np.isin(idp, pliki_test)
    if m_test.sum() == 0:
        log("UWAGA: zadne okno nie pasuje do plikow testowych z 05.")
        log("Czy okna_dane.npz jest to samo, na ktorym trenowano? Przelicz 01->02->05.")
        log.zapisz()
        return

    Xte = (X[m_test] - srednia) / odchyl
    prob_test = model.predict(Xte, verbose=0).ravel()
    y_test = y[m_test]
    idp_test = idp[m_test]
    odch_test = odch[m_test]

    auc = roc_auc_score(y_test, prob_test) if len(np.unique(y_test)) > 1 else float("nan")
    log(f"AUROC na tescie (ta sama siec co w 05): {auc:.3f}\n")

    dt_okna = krok_czasu_okna()
    log(f"Jedno okno do nastepnego = {dt_okna:.2f} s")
    log(f"Horyzont H={HORYZONT_H} okien = {HORYZONT_H*dt_okna:.1f} s w przyszlosc")
    log("(UWAGA: to ZALOZONE wyprzedzenie - wynika z konstrukcji etykiety,")
    log(" ktora z definicji patrzy H okien w przod; nie jest mierzone z danych.)\n")

    # ============================================================
    # WIDOK 3: ocena per prog decyzyjny
    # ============================================================
    log("=" * 64)
    log("OCENA PER PROG DECYZYJNY")
    log("=" * 64)
    log(f"{'prog':>5} | {'alarmy':>7} | {'trafione':>8} | {'falszywe':>8} | "
        f"{'przeocz.':>8} | {'wyprzedz.[s]':>12}")
    log("-" * 64)

    for prog in PROGI_ALARMU:
        alarm = prob_test >= prog
        trafione = int((alarm & (y_test == 1)).sum())
        falszywe = int((alarm & (y_test == 0)).sum())
        przeocz = int((~alarm & (y_test == 1)).sum())

        # zalozone wyprzedzenie: H okien * dt (etykieta patrzy H okien w przod)
        wyprzedzenie_s = HORYZONT_H * dt_okna if trafione > 0 else 0
        log(f"{prog:>5.1f} | {int(alarm.sum()):>7} | {trafione:>8} | "
            f"{falszywe:>8} | {przeocz:>8} | {wyprzedzenie_s:>12.1f}")

    # ============================================================
    # WIDOK 2 (tabela): werdykt per plik (przy srodkowym progu)
    # ============================================================
    prog_glowny = PROGI_ALARMU[len(PROGI_ALARMU) // 2]
    log(f"\n{'='*64}")
    log(f"WERDYKT PER AMORTYZATOR (prog alarmu = {prog_glowny})")
    log("=" * 64)
    log(f"{'plik':>8} | {'% alarmow':>10} | werdykt")
    log("-" * 64)
    for p in sorted(pliki_test):
        m = idp_test == p
        proc_alarm = 100 * (prob_test[m] >= prog_glowny).mean()
        werdykt = "WYMAGA UWAGI" if proc_alarm > 20 else "zdrowy"
        log(f"plik #{p:>3} | {proc_alarm:>9.1f}% | {werdykt}")

    # ============================================================
    # WIDOK 1: wykres symulacji na zywo (jeden plik testowy)
    # ============================================================
    # wybierz do wykresu plik testowy z NAJWIEKSZA liczba alarmow (najciekawszy)
    prog_demo = PROGI_ALARMU[len(PROGI_ALARMU) // 2]
    najwiecej, plik_demo = -1, sorted(pliki_test)[0]
    for p in pliki_test:
        mm = idp_test == p
        ile = int((prob_test[mm] >= prog_demo).sum())
        if ile > najwiecej:
            najwiecej, plik_demo = ile, p
    m = idp_test == plik_demo
    odch_p = odch_test[m]
    prob_p = prob_test[m]
    y_p = y_test[m]
    czas = np.arange(len(odch_p)) * dt_okna

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # gorny: odchylka w czasie + co bylo prawdziwa etykieta 1
    ax1.plot(czas, odch_p, color="#5F5E5A", label="biezaca odchylka")
    ax1.scatter(czas[y_p == 1], odch_p[y_p == 1], s=20, color="#D85A30",
                zorder=3, label="faktyczna 'nadchodzi' (prawda)")
    ax1.axhline(PROG_WZGLEDNY, color="#888780", linestyle=":", label=f"prog etykiet {PROG_WZGLEDNY}")
    ax1.set_ylabel("odchylka wzgledna")
    ax1.set_title(f"Plik #{plik_demo}: co sie dzieje w danych")
    ax1.legend(fontsize=8)

    # dolny: prawdopodobienstwo alarmu sieci + progi
    ax2.plot(czas, prob_p, color="#378ADD", label="alarm sieci (prawdop.)")
    for prog in PROGI_ALARMU:
        ax2.axhline(prog, alpha=0.4, linewidth=0.8, linestyle="--")
    ax2.fill_between(czas, 0, 1, where=(prob_p >= prog_glowny),
                     alpha=0.15, color="#D85A30", label=f"alarm (prog {prog_glowny})")
    ax2.set_ylabel("prawdop. alarmu")
    ax2.set_xlabel("czas [s]")
    ax2.set_title("Co przewiduje siec (system wczesnego ostrzegania)")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    sciezka_png = sciezka_wyniku("zastosowanie.png")
    plt.savefig(sciezka_png, dpi=120)
    plt.close(fig)
    log(f"\nWykres zapisany do: {sciezka_png} (plik #{plik_demo})")
    log.zapisz()


if __name__ == "__main__":
    main()
