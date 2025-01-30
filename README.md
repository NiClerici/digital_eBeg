# eBew: Baugesuchs Portal

## Übersicht
eBew ist ein Portal zur Verwaltung und Analyse von Baugesuchen. Es ermöglicht die automatische Erkennung und Extraktion von Informationen aus Bauplänen. Das Projekt basiert auf Frameworks wie Detectron2 und PaddleOCR, um Bildverarbeitungs- und OCR-Funktionen bereitzustellen.

## Hauptskripte

### 1. `Inferenz.py`
Dieses Skript ist dafür zuständig, vortrainierte Modelle zu verwenden, um Baupläne zu analysieren und Objekte wie Gebäude, Straßen oder Parzellennummern zu erkennen. Die Skript-Funktionen sind:

- **Laden und Konfigurieren eines vortrainierten Mask R-CNN-Modells** aus dem Detectron2-Framework.
- **Anpassung der Schwellenwerte und Modelleinstellungen** für eine benutzerdefinierte Anzahl von Klassen.
- **Visualisierung der Analyseergebnisse** mit einer detaillierten Ausgabe aller erkannten Objekte und deren Wahrscheinlichkeit.

### Voraussetzungen:
- Ein vortrainiertes Modell (`model_final.pth`) muss im angegebenen Dateipfad vorhanden sein.
- Erforderliche Datensätze mit benutzerdefinierten Klassen sollten eingerichtet sein.

---

### 2. `Train_32x8d_010125.py`
Dieses Skript dient zum Trainieren einer benutzerdefinierten Detektions- und Segmentierungspipeline. Es umfasst folgende wichtige Funktionen und Klassen:

- **Dataset-Registrierung und -Kombination**: Es ermöglicht das Hinzufügen mehrerer Datensätze und deren Augmentierung.
- **WandbTrainer-Klasse**: Eine angepasste Trainingsklasse, die auf Detectron2 basiert und Workflows wie Logging unterstützt (z. B. via Weights & Biases).
- **Fortgeschrittene Augmentierungsmethoden** für Bilder zur Optimierung des Trainings.

### Voraussetzungen:
- Geladene Datensätze im richtigen Format.
- Eine korrekte Konfigurationsdatei für das Detectron2-Modell.
- Python-Pakete wie `numpy`, `detectron2`, und `wandb`.

---

### 3. `detect_OCR.py`
Dieses Skript kombiniert die Funktionalitäten der Objekterkennung (Detectron2) und einer OCR (PaddleOCR), um spezifische Daten (z. B. Masslinien oder Parzellennummern) direkt aus Bauplänen zu extrahieren.

- **Benutzerdefinierte Bildvorverarbeitung**: Skalierung und Schärfen der Bilder zur Verbesserung der OCR-Erkennung.
- **Objekterkennung und Textauslesung**: Lokalisierung spezifischer Bereiche im Bild und Extraktion von Text in einem hohen Maß an Genauigkeit.
- **Integration von PaddleOCR**: Erkennung von Textknoten innerhalb relevanter Klassen.

### Voraussetzungen:
- PaddleOCR muss installiert und richtig konfiguriert sein.
- Temporäres Schreibrecht in das `/tmp`-Verzeichnis für Zwischenspeicher.

---

## Installation

1. **Repository klonen**:
    ```bash
    git clone https://github.com/NiClerici/eBew.git
    cd eBew
    ```

2. **Virtuelle Umgebung erstellen**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate     # Für Windows: venv\Scripts\activate
    ```

3. **Abhängigkeiten installieren**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Modelldateien hinzufügen**:
    - Lade ein vortrainiertes Detectron2-Modell (z. B. `mask_rcnn_X_101_32x8d_FPN_3x.yaml`) herunter.
    - Stelle sicher, dass die Datei `model_final.pth` vorhanden und die Konfiguration entsprechend angepasst ist.

## Nutzung

### Skripte ausführen:

1. **Objekterkennung (`Inferenz.py`)**:
    - Stelle sicher, dass dein Bildpfad in den Konfigurationen des Skripts korrekt angegeben ist.
    - Starte das Skript:
        ```bash
        python Inferenz.py
        ```

2. **Training (`Train_32x8d_010125.py`)**:
    - Lade deine Datensätze und stelle sicher, dass alles richtig registriert ist.
    - Passe die Trainingsparameter an und führe das Skript aus:
        ```bash
        python Train_32x8d_010125.py
        ```

3. **Text-Erkennung und OCR (`detect_OCR.py`)**:
    - Erstelle eine Input-Bilddatei und führe das Skript mit dem Pfad des Bildes aus:
        ```bash
        python detect_OCR.py
        ```

---

## Projektstruktur

- **`Inferenz.py`**: Analysemodul für Baupläne.
- **`Train_32x8d_010125.py`**: Trainingsskript für benutzerdefinierte Modelle.
- **`detect_OCR.py`**: Modul zur kombinierten Objekt- und Texteerkennung.
- **`requirements.txt`**: Liste aller erforderlichen Python-Abhängigkeiten.
- **`/src/`**: Hauptverzeichnis mit Modellen, Bildern und Konfigurationen.

---

## Benötigte Abhängigkeiten
Die wichtigsten Python-Bibliotheken, die für dieses Projekt benötigt werden, sind:
- `detectron2`
- `paddleocr`
- `numpy`
- `opencv-python`
- `matplotlib`

Stelle sicher, dass sie mit `pip install -r requirements.txt` installiert werden.

---

## Lizenz
Dieses Projekt ist unter der MIT-Lizenz verfügbar. Weitere Informationen findest du in der `LICENSE`-Datei des Repositories.

---

## Kontakt
Für Fragen und Feedback wende dich bitte an [Nico Clerici](mailto:nico.clerici@example.com).