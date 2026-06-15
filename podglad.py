import numpy as np
import matplotlib.pyplot as plt

PLIK = "okna_etykiety.npz"

# ile okien narysowac na wykresie
ILE_OKIEN_NA_WYKRESIE = 4

def main():
    dane = np.load(PLIK, allow_pickle=True)
    print(f"Plik: {PLIK}")
    print("Co siedzi w srodku (klucze):", list(dane.keys()))
    print()
    X = dane["X"]
    print(f"X - same dane okien")
    print(f"  ksztalt: {X.shape}")
    print(f"  to znaczy: {X.shape[0]} okien, kazde {X.shape[1]} probek x {X.shape[2]} kolumny")
    print(f"  kolumny to: [0]=sila[kN], [1]=przemieszczenie[mm], [2]=predkosc[mm/s]")
    print()

    if "id_pliku" in dane:
        idp = dane["id_pliku"]
        print(f"id_pliku - z ktorego pliku pochodzi kazde okno")
        print(f"  unikalne pliki: {np.unique(idp)}")
        print(f"  okien na plik: {[int((idp==i).sum()) for i in np.unique(idp)]}")
        print()

    if "nazwy_plikow" in dane:
        print("nazwy_plikow:", list(dane["nazwy_plikow"]))
        print()

    if "y" in dane:
        y = dane["y"]
        print(f"y - etykiety (0=poprawne, 1=odchylka)")
        print(f"  poprawne: {(y==0).sum()}, odchylki: {(y==1).sum()}")
        print()


    print("=== Pierwsze okno, pierwsze 5 probek ===")
    print("  sila      przem     predkosc")
    for wiersz in X[0][:5]:
        print(f"  {wiersz[0]:8.3f}  {wiersz[1]:8.3f}  {wiersz[2]:8.3f}")
    print()

    # --- 3. Wykresy kilku okien ---
    n = min(ILE_OKIEN_NA_WYKRESIE, len(X))
    fig, osie = plt.subplots(n, 2, figsize=(11, 2.5 * n))
    if n == 1:
        osie = osie.reshape(1, 2)

    for i in range(n):
        okno = X[i]
        # lewy wykres: sila i przemieszczenie w czasie
        ax = osie[i, 0]
        ax.plot(okno[:, 0], label="sila [kN]")
        ax.plot(okno[:, 1], label="przem [mm]")
        ax.set_title(f"Okno {i} - przebieg w czasie")
        ax.set_xlabel("nr probki w oknie")
        ax.legend(fontsize=8)

        # prawy wykres: petla sila-predkosc (charakterystyka tlumienia)
        ax = osie[i, 1]
        ax.plot(okno[:, 2], okno[:, 0], ".-")
        ax.set_title(f"Okno {i} - sila vs predkosc")
        ax.set_xlabel("predkosc [mm/s]")
        ax.set_ylabel("sila [kN]")

    plt.tight_layout()
    plt.savefig("podglad.png", dpi=120)
    print("Wykresy zapisane do: podglad.png")


if __name__ == "__main__":
    main()
