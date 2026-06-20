# TODO / Roadmap

## Webapp GUI

Ziel: Eine Web-GUI, die das Arbeiten mit dem Toolchain über Browser ermöglicht
(aktuell ist `src/webapp/` ein dünner Wrapper um die CLI `python run.py ...`).

### Kernkonzept: Projekte
Alles wird unter **Projekten** organisiert. Ein Projekt bündelt die zusammen-
gehörigen Artefakte eines Experiments/Vorhabens. Ein Projekt soll enthalten können:

- [ ] **Modelle** — die (Float-)Basismodelle
- [ ] **Quantisierte Modelle** — aus den Basismodellen abgeleitete, quantisierte Varianten
- [ ] **Trainingsdaten** — Datensätze, die einem Projekt zugeordnet sind
- [ ] **Hardware-Modelle** — Hardware-/Target-Beschreibungen für Deployment/Synthese

### Aufgaben
- [x] Datenmodell für Projekte definieren (Schema + Persistenz) — `src/webapp/projects.py` (SQLite, stdlib)
- [x] Beziehungen abbilden: Projekt → Modelle → quantisierte Modelle; Projekt → Trainingsdaten; Projekt → Hardware-Modelle (`parent_id` für Lineage)
- [x] API-Endpunkte für CRUD auf Projekten und deren Artefakten (`/api/projects...` in `server.py`)
- [x] GUI-Ansichten: Projektübersicht + Projekt-Detail mit den vier Artefakttypen (`static/index.html`)
- [ ] Anbindung an bestehende Jobs (Training, QAT, Export) im Projektkontext
- [ ] Artefakte direkt aus Job-Ergebnissen anlegen (z. B. Export → quant_model automatisch ans Projekt hängen)
- [ ] Datei-Upload / Datei-Browser statt nur Pfad-Eingabe für Artefakte

## Experiment-Builder (exp_config.yaml in der GUI erstellen)

Netzbeschreibung (Modell-YAML) öffnen/wählen → Dataset wählen → Training
konfigurieren → daraus entsteht eine `exp_config.yaml`, die Netz + Dataset +
Training zusammenfasst.

### Aufgaben
- [x] Builder-Logik: spec → exp_config-dict → YAML (`src/webapp/expbuilder.py`)
- [x] Netzbeschreibung öffnen (Modell-YAML-Inhalt anzeigen)
- [x] Dataset-Liste aus der Registry (Fallback ohne torch)
- [x] API: `/api/datasets`, `/api/model-content`, `/api/experiments/preview`, `/api/experiments`
- [x] GUI-Panel: Felder + Preview + Save (neue Config erscheint im Train/QAT-Dropdown)
- [x] Verifiziert: generierte YAML lädt über den projekteigenen `ConfigReader`
- [ ] Netzbeschreibung selbst in der GUI bauen/bearbeiten (Layer-Liste editieren),
      nicht nur referenzieren
- [ ] Experiment direkt aus dem Builder einem Projekt zuordnen

## Transformationen tracken

Ziel: Den Weg vom Float-Modell bis zur Hardware lückenlos nachvollziehbar machen.
Heute ist die Provenance nur fragmentarisch (`info.json` im Bundle, SQLite
`experiment_db` fürs Training) — es fehlt eine durchgehende Kette mit Parametern
und Hashes.

### Die Transformationskette (Ist-Stand im Code)
1. **Float-Modell** — `build_network()` aus Architektur-YAML (reines PyTorch)
2. **Quantisierung (QAT)** — `run.py qat`, `src/core/qat_schedule.py`: progressive
   Bit-Breiten-Stufen (z. B. w8a8 → w4a4 → w2a2) mit Rollback bei `--min-acc`
   → quantisiertes Brevitas-Modell
3. **Export → QONNX** — `run.py export`, `src/export/qonnx_export.py`: QONNX-ONNX
   (optional Input-Quant), optional `--bundle` (qonnx + build.py + yaml + weights + info.json)
4. **FINN Dataflow-Build** (im FINN-Docker) — `src/finn`: streamline →
   MultiThreshold (Integer-Schwellwerte, Dump für FloPoCo-Encoder) →
   minimize_bit_width → HLS → synth → stitched IP → OOC-Synth → Bitfile
5. **Hardware-Artefakt** — Bitfile + PYNQ-Treiber + Deployment-Package (+ gedumpte
   Integer-Thresholds & -Gewichte)

### Aufgaben
- [ ] Jede Transformation als Datensatz erfassen: Typ, Eingangs-Artefakt, Ausgangs-Artefakt,
      Parameter (Bit-Schedule, Input-Quant, FPGA-Part/Board, synth-Level), Zeitstempel
- [ ] Artefakte über Content-Hash identifizieren → reproduzierbare Verkettung
- [ ] Durchgehende Lineage abbilden: Float → QAT-Stufen → QONNX → FINN-Build → Bitfile
- [ ] Bestehende Quellen vereinheitlichen (`info.json`-Bundle + `experiment_db`) statt parallel
- [ ] In der GUI als Transformations-/Lineage-Graph pro Projekt darstellen
- [ ] FINN-Build-Schritte (streamline/threshold/minimize/synth) als Stufen mit Status festhalten
- [ ] Klären/dokumentieren: was genau ist das finale HW-Deliverable je synth-Level
      (estimate = Reports, ip = stitched IP, bitfile = Bitstream + Deployment-Package)

## Transformationspfad kennen → Fehler beheben (explorativ)

Kernidee: Wenn der Weg von A nach B (jede Transformationsstufe, ihre Ein-/Ausgänge
und Parameter) erfasst ist, lässt sich ein Fehler an genau der Stufe lokalisieren,
an der er auftritt — und die Reparatur kann explorativ Alternativen durchprobieren,
bis der Pfad wieder durchläuft.

Hintergrund: Ab dem QONNX-Export sind alle HW-Transformationen ONNX→ONNX-Graph-
Operationen (streamline, MultiThreshold-Fold, minimize_bit_width, HLS, synth). Jede
Stufe hat ein wohldefiniertes Eingangs- und Ausgangs-Artefakt — das macht sie
einzeln nachvollziehbar und wiederholbar.

### Aufgaben / offene Fragen
- [ ] Jede Stufe als reproduzierbaren Schritt erfassen (Eingang, Ausgang, Parameter,
      Status ok/fehlgeschlagen + Fehlermeldung) — baut auf dem Transformations-Tracking auf
- [ ] Fehler der Stufe zuordnen, an der er auftritt (statt nur „Build fehlgeschlagen")
- [ ] Ab der fehlerhaften Stufe neu starten können, ohne die ganze Kette zu wiederholen
- [ ] Explorative Reparatur: Alternativen für die fehlerhafte Stufe automatisch durchprobieren
      (z. B. andere Bit-Breite, anderer FPGA-Part, anderer Streamline-/Schritt-Pfad)
- [ ] Bekannte Fehler→Fix-Muster sammeln und wiederverwenden (Wissensbasis)
- [ ] Schnelle Signale nutzen: `estimate`-Reports als billiges Feedback vor vollem Bitfile-Build
- [ ] In der GUI: den Transformationspfad mit Status je Stufe zeigen, fehlerhafte Stufe markieren

## FINN (Docker) automatisiert starten + Synthese fahren

Ziel: Den FINN-Build inkl. Synthese auf Knopfdruck / automatisiert starten, ohne
manuell ins FINN-Docker zu wechseln. FINN läuft nur in seinem Docker-Image (braucht
Vivado/Vitis), darum muss der Lauf containerisiert orchestriert werden.

Ist-Stand: `JobManager` (`src/webapp/jobs.py`) startet beliebige Kommandos als
Subprozess und streamt Logs; `server.py` ruft bereits `./run-docker.sh python build.py`
im Bundle-Ordner auf. `export --bundle` erzeugt den selbstständigen Build-Ordner
(qonnx + build.py + info.json). Es fehlt die eigentliche Docker-Orchestrierung.

### Aufgaben / offene Fragen
- [ ] Docker-Aufruf kapseln: FINN-Image starten, Bundle-Ordner als Volume mounten,
      `build.py` ausführen, Build-Outputs zurückschreiben
- [ ] Voraussetzungen prüfen/melden (Docker vorhanden, FINN-Image vorhanden/pullbar,
      Vivado/Vitis-Lizenz erreichbar) statt kryptischer Fehler
- [ ] synth-Level durchreichen (estimate / ip / bitfile) und die passenden Outputs erwarten
- [ ] Build als Job über `JobManager` führen: Live-Logs, Stop, Status, Rückgabecode
- [ ] Lange Laufzeiten der Synthese handhaben (Stunden): Hintergrundlauf, Wiederanlauf,
      Fortschritt/ETA aus den FINN-Schritten ableiten
- [ ] Outputs einsammeln und als Artefakte ans Projekt hängen (Reports, stitched IP,
      Bitfile, PYNQ-Treiber, Deployment-Package, Thresholds/Gewichte)
- [ ] An Tracking + Fehlerbehebung koppeln: Build-Schritte als Stufen mit Status,
      fehlerhafte Stufe markieren (siehe Abschnitt oben)
- [ ] Später: Remote-Worker-Variante (das `JobManager`-Interface ist dafür schon vorgesehen)

## Jobs auf verschiedene Rechner verteilen

Ziel: Jobs (Training, QAT, Export, FINN-Synthese) nicht nur lokal, sondern auf
verschiedenen Rechnern starten können — z. B. Training auf einer GPU-Maschine,
FINN-Synthese auf einem Rechner mit Vivado-Lizenz.

Ist-Stand: `JobManager` (`src/webapp/jobs.py`) ist laut Docstring bewusst so gebaut,
dass „remote worker agents" dasselbe Interface anbieten — der Controller soll lokale
und entfernte Jobs gleich behandeln. Heute läuft alles als lokaler Subprozess.

### Aufgaben / offene Fragen
- [ ] Worker-Konzept: jeder Rechner stellt das `JobManager`-Interface bereit
      (start / stop / status / log streamen)
- [ ] Worker registrieren/auflisten (Adresse, Fähigkeiten: GPU? Vivado? Docker?)
- [ ] Job-Routing: Job an einen passenden Worker schicken (manuell wählbar + automatisch
      nach Anforderungen, z. B. FINN → Worker mit Vivado)
- [ ] Logs/Status entfernter Jobs einheitlich in die GUI streamen (wie lokale Jobs)
- [ ] Artefakte zwischen Rechnern transportieren (Checkpoints, QONNX, Bundle, Build-Outputs)
- [ ] Authentifizierung/Transport festlegen (z. B. HTTP-API der Worker, SSH, o. Ä.)

## ONNX-Transformationen interaktiv ansehen

Ziel: Sehen, wie sich der ONNX-Graph durch eine Transformation verändert — also den
Graph vor und nach jedem Schritt interaktiv vergleichen, statt nur am Endergebnis.

Hintergrund: Ab dem QONNX-Export ist jeder HW-Schritt eine ONNX→ONNX-Transformation
(streamline, MultiThreshold-Fold, minimize_bit_width, …). Wenn jeder Schritt seinen
ONNX-Zwischenstand ablegt, lässt sich die Veränderung Schritt für Schritt zeigen.

### Aufgaben / offene Fragen
- [ ] Pro Transformationsschritt den ONNX-Zwischenstand speichern (Snapshot je Stufe)
- [ ] ONNX-Graph in der GUI rendern (Knoten, Tensoren, Datentypen/Bit-Breiten)
- [ ] Vorher/Nachher-Vergleich je Schritt: was wurde gefaltet/entfernt/zusammengelegt
      (Diff hervorheben)
- [ ] Durch die Schritte navigieren (Slider/Timeline entlang des Transformationspfads)
- [ ] Knoten-Details zeigen (z. B. MultiThreshold-Schwellen, Gewichte, Datentypen)
- [ ] An Tracking + Fehlerbehebung koppeln: am fehlerhaften Schritt direkt den Graph ansehen
- [ ] Klären: bestehenden Viewer (z. B. Netron) einbetten vs. eigene Darstellung
