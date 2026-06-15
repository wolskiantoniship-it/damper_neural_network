# Diagnostyka amortyzatora - siec neuronowa (CNN1D)

System wczesnego ostrzegania: na podstawie krotkiego okna sygnalu
(sila / przemieszczenie / predkosc) siec przewiduje, czy za H okien
NADEJDZIE odchylka sily od wzorca liniowego F = C*v.

## Wymagania

- Python 3.10+
- `numpy`, `matplotlib`, `scikit-learn`, `tensorflow` (lub `tensorflow-cpu`)

Instalacja:

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

## Struktura

- `parametry.py` - **jedno zrodlo prawdy**: stale (prog, horyzont, wzorzec),
  funkcje wspolne (`biezace_odchylki`, `etykiety_predykcyjne`), folder wynikow
  i logger. Jesli chcesz zmienic prog/horyzont - zmieniasz TYLKO tutaj.
- `dane/` - wejsciowe pliki CSV (eksport z DAQ; polskie liczby z przecinkiem).
- `wyniki/` - tworzony automatycznie; tu laduja WSZYSTKIE wyniki
  (pliki .npz, model .keras, wykresy .png oraz logi .txt z kazdego skryptu).

## Kolejnosc uruchamiania

```bash
python 01_parsowanie.py     # CSV -> okna; zapisuje wyniki/okna_dane.npz
python 02_etykietuj.py      # nadaje etykiety predykcyjne -> wyniki/okna_etykiety.npz
python 03_podglad_all.py    # (opcjonalne) diagnoza: okna, etykiety, prog
python 05_siec.py           # trenuje CNN1D; zapisuje model + parametry + wyniki.png
python 06_zastosowanie.py   # demonstracja systemu na plikach testowych

python 04_horyzonty.py      # (osobna analiza) AUROC vs horyzont H, srednia z powtorzen
```

Skrypty 03 i 04 sa pomocnicze/analityczne i nie sa wymagane do otrzymania
modelu - ale 04 dokumentuje stabilnosc wyniku (wazne przy malym zbiorze).

## Opis skryptow

| Plik | Co robi |
|------|---------|
| `01_parsowanie.py` | Wczytuje CSV, liczy predkosc (gradient), tnie na okna 20x3 z zakladka 50%. |
| `02_etykietuj.py` | Etykieta okna n = 1, jesli odchylka okna n+H przekracza prog. Pomija konce plikow. |
| `03_podglad_all.py` | Diagnoza: podglad okien, balans klas, wplyw progu na balans. |
| `04_horyzonty.py` | AUROC sieci vs horyzont H, powtarzane z roznymi podzialami (srednia +- odchylenie). |
| `05_siec.py` | Trenuje CNN1D z podzialem PO PLIKACH, skalowanie z-score z treningu. Zapisuje model + metryki. |
| `06_zastosowanie.py` | Wczytuje gotowy model (nie trenuje) i pokazuje dzialanie jako system wczesnego ostrzegania. |

## Wazne uwagi metodologiczne

- **Podzial po plikach, nie po oknach** - okna z jednego biegu sa podobne,
  wiec test musi zawierac CALE, niewidziane pliki (inaczej wynik jest zawyzony).
- **Skalowanie liczone tylko z treningu** - test nie "podglada" swoich statystyk.
- **Maly zbior testowy** - przy 18 plikach test to ~3 pliki. Pojedynczy AUROC
  moze byc szczesciem przy danym podziale; dlatego 04_horyzonty.py powtarza
  trening z roznymi ziarnami i pokazuje rozrzut.
- **"Wyprzedzenie" w 06 jest ZALOZONE**, nie mierzone - wynika z definicji
  etykiety (patrzy H okien w przod), wiec poprawny alarm wyprzedza zdarzenie
  o ~H*dt sekund z konstrukcji.
- **Dwie populacje plikow** - czesc plikow ma 0% klasy 1, czesc 30-50%.
  Warto to nazwac w opisie: model rozroznia zachowanie zgodne vs odbiegajace
  od wzorca w obrebie tej proby, a nie ciagla progresje uszkodzenia w czasie.
