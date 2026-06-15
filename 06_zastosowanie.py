"""
05_zastosowanie.py
==================
Co robi (po ludzku):
Pokazuje, jak wytrenowana siec dzialalaby W PRAKTYCE jako system wczesnego
ostrzegania. UZYWA GOTOWEJ sieci z 03_siec.py (nie trenuje od nowa!), wiec
demonstruje DOKLADNIE ten model, ktorego metryki podajesz w artykule.

Najpierw uruchom 03_siec.py - on zapisuje:
  - model_amortyzator.keras  (wytrenowana siec)
  - model_parametry.npz       (skalowanie + ktore pliki sa testowe)
05_zastosowanie.py wczytuje oba i przejezdza pliki testowe okno po oknie.

Liczy TRZY rzeczy:
  WIDOK 1 - symulacja na zywo (wykres alarmow w czasie),
  WIDOK 2 - wyprzedzenie czasowe (ile sekund wczesniej ostrzega),
  WIDOK 3 - ocena calego amortyzatora (zdrowy / wymaga uwagi).

WAZNE - zsynchronizuj te parametry z 02_etykietuj.py:
  HORYZONT_H, PROG_WZGLEDNY, C_WZORZEC, EPS_KN

Wymaga: okna_dane.npz, model_amortyzator.keras, model_parametry.npz

Uruchomienie:
    python 05_zastosowanie.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow import keras

# === PARAMETRY - ZSYNCHRONIZUJ Z 02_etykietuj.py ===
PLIK_DANE = "okna_dane.npz"
PLIK_MODEL = "model_amortyzator.keras"
PLIK_PARAM = "model_parametry.npz"
C_WZORZEC = 0.00278
EPS_KN = 0.05
HORYZONT_H = 10          # <-- TAKI SAM jak w 02_etykietuj.py
PROG_WZGLEDNY = 4.0      # <-- TAKI SAM jak w 02_etykietuj.py (Twoj wybor: 4.0)

# progi decyzyjne sieci do porownania (osobne od progu etykiet!)
PROGI_ALARMU = [0.3, 0.5, 0.7]

# czestotliwosc probkowania i dlugosc okna - do przeliczenia okien na SEKUNDY
HZ = 10.0                # probkowanie DAQ [Hz]
DLUGOSC_OKNA = 20        # probek na okno (z 01_parsuj.py)
ZAKLADKA = 0.5           # zakladka okien (z 01_parsuj.py)


# ---- odchylka i etykiety (identyczne jak w 02) ----
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


def krok_czasu_okna():
    """Ile SEKUND dzieli poczatki dwoch sasiednich okien."""
    krok_probek = DLUGOSC_OKNA * (1 - ZAKLADKA)
    return krok_probek / HZ


def main():
    dane = np.load(PLIK_DANE, allow_pickle=True)
    X = dane["X"].astype("float32")
    idp = dane["id_pliku"]

    odch = biezace_odchylki(X, C_WZORZEC)
    y = etykiety_predykcyjne(odch, idp, HORYZONT_H, PROG_WZGLEDNY)
    maska = y >= 0
    X, y, idp, odch = X[maska], y[maska], idp[maska], odch[maska]

    # --- WCZYTUJEMY gotowa siec i parametry (zamiast trenowac) ---
    try:
        model = keras.models.load_model(PLIK_MODEL)
        param = np.load(PLIK_PARAM, allow_pickle=True)
    except (OSError, IOError, ValueError) as e:
        print("Nie moge wczytac modelu/parametrow.")
        print(f"Blad: {e}")
        print("\nNajpierw uruchom 03_siec.py - on tworzy:")
        print("  model_amortyzator.keras  oraz  model_parametry.npz")
        return

    srednia = param["srednia"]
    odchyl = param["odchyl"]
    pliki_test = param["pliki_testowe"]
    print(f"Wczytano gotowa siec. Pliki testowe (z 03_siec.py): {sorted(pliki_test)}")

    # bierzemy DOKLADNIE te pliki, ktore byly testowe w 03 (siec ich nie widziala)
    m_test = np.isin(idp, pliki_test)
    if m_test.sum() == 0:
        print("UWAGA: zadne okno nie pasuje do plikow testowych z 03.")
        print("Czy okna_dane.npz jest to samo, na ktorym trenowano? Przelicz 01->02->03.")
        return

    Xte = (X[m_test] - srednia) / odchyl
    prob_test = model.predict(Xte, verbose=0).ravel()
    y_test = y[m_test]
    idp_test = idp[m_test]
    odch_test = odch[m_test]

    auc = roc_auc_score(y_test, prob_test) if len(np.unique(y_test)) > 1 else float("nan")
    print(f"AUROC na tescie (ta sama siec co w 03): {auc:.3f}\n")

    dt_okna = krok_czasu_okna()
    print(f"Jedno okno do nastepnego = {dt_okna:.2f} s")
    print(f"Horyzont H={HORYZONT_H} okien = {HORYZONT_H*dt_okna:.1f} s w przyszlosc\n")

    # ============================================================
    # WIDOK 3: ocena calego amortyzatora + wyprzedzenie - dla kilku progow
    # ============================================================
    print("=" * 64)
    print("OCENA PER PROG DECYZYJNY")
    print("=" * 64)
    print(f"{'prog':>5} | {'alarmy':>7} | {'trafione':>8} | {'falszywe':>8} | "
          f"{'przeocz.':>8} | {'wyprzedz.[s]':>12}")
    print("-" * 64)

    for prog in PROGI_ALARMU:
        alarm = prob_test >= prog
        trafione = int((alarm & (y_test == 1)).sum())
        falszywe = int((alarm & (y_test == 0)).sum())
        przeocz = int((~alarm & (y_test == 1)).sum())

        # wyprzedzenie: H okien * dt (etykieta patrzy H okien w przod,
        # wiec poprawny alarm wyprzedza odchylke o ~H okien czasu)
        wyprzedzenie_s = HORYZONT_H * dt_okna if trafione > 0 else 0
        print(f"{prog:>5.1f} | {int(alarm.sum()):>7} | {trafione:>8} | "
              f"{falszywe:>8} | {przeocz:>8} | {wyprzedzenie_s:>12.1f}")

    # ============================================================
    # WIDOK 2 (tabela): werdykt per plik (przy srodkowym progu)
    # ============================================================
    prog_glowny = PROGI_ALARMU[len(PROGI_ALARMU) // 2]
    print(f"\n{'='*64}")
    print(f"WERDYKT PER AMORTYZATOR (prog alarmu = {prog_glowny})")
    print("=" * 64)
    print(f"{'plik':>8} | {'% alarmow':>10} | werdykt")
    print("-" * 64)
    for p in sorted(pliki_test):
        m = idp_test == p
        proc_alarm = 100 * (prob_test[m] >= prog_glowny).mean()
        werdykt = "WYMAGA UWAGI" if proc_alarm > 20 else "zdrowy"
        print(f"plik #{p:>3} | {proc_alarm:>9.1f}% | {werdykt}")

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
    plt.savefig("zastosowanie.png", dpi=120)
    print(f"\nWykres zapisany do: zastosowanie.png (plik #{plik_demo})")


if __name__ == "__main__":
    main()
