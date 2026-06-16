# BioSafe

Keystroke-dynamics biometric authentication — verify identity by how you type, not what you type.

## Quick Start

```
git clone https://github.com/Gogo1206/BioSafe.git
cd BioSafe
pip install -r requirements.txt
python src/interface/run.py
```

## How It Works

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Keystroke       │     │  Feature Vector   │     │  Random Forest   │
│  Capture         │ ──► │  4 press intervals│ ──► │  Prediction      │
│  (pynput)        │     │  + 4 hold delays  │     │  user/others/ai  │
└──────────────────┘     └───────────────────┘     └──────────────────┘
```

1. User types a 4-digit PIN — `pynput` records press and release timestamps
2. 8 features extracted: 4 inter-key intervals + 4 key hold durations
3. Random Forest classifier trained on CSV data in `tmp/data/`
4. Real-time prediction on new keystrokes via GUI or terminal

## Project Structure

```
src/
├── type.py              → Standalone keystroke capture tool
├── graph.py             → Typing pattern visualization
├── randomForest.py      → Model training + evaluation + tree viz
├── data/
│   ├── data.py          → Synthetic "others" data generator
│   ├── ai.py            → Synthetic AI brute-force data generator
│   └── run.py           → Terminal keystroke capture + prediction
├── interface/
│   ├── run.py           → Tkinter GUI login + prediction
│   └── window.py        → Basic Tkinter login stub
└── old/                 → Earlier versions and ML experiments
```

## Usage

| Command | Purpose |
|---------|---------|
| `python src/interface/run.py` | GUI application with login screen |
| `python src/data/run.py` | Terminal demo with AI brute-force detection |
| `python src/type.py` | Raw keystroke data collection |
| `python src/randomForest.py` | Train model and view decision trees |
| `python src/graph.py` | Plot typing patterns from CSV |

Data stored in `tmp/data/` (gitignored). Each CSV file = one identity label.

## Dependencies

Python 3, scikit-learn, pynput, tkinter, matplotlib, seaborn, numpy, pandas.

See [`requirements.txt`](requirements.txt).

## Limitations

- **4 keystrokes is minimal signal** — 7 real features is borderline for biometric auth. Consider longer passphrases or continuous auth.
- **Multi-class framing is wrong for authentication** — should be one-class ("is this the user?") not "which known category?" See roadmap for details.
- **Session-internal evaluation only** — accuracy numbers are inflated. Real-world use would need cross-session data collection.

## Future Work

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for prioritized improvements including:
- One-class authentication architecture
- Cross-session enrollment and evaluation
- FAR/FRR biometric metrics
- Per-key modeling and score fusion

## License

Educational project — use at your own risk. Not production-ready authentication.
