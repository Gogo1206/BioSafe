# Project Instructions

## Tech Stack
- Python 3 (no version pin)
- scikit-learn (RandomForestClassifier)
- pynput (keyboard monitoring)
- tkinter (GUI)
- matplotlib + seaborn (visualization)
- numpy + pandas (data manipulation)

## Code Style
- File naming: camelCase (`randomForest.py`, not `random_forest.py`)
- Imports at top of file, stdlib first then third-party
- Flat structure — no packages, no `__init__.py`
- All scripts use `if __name__ == "__main__"` guard — safe to import
- No type hints, no docstrings, no linting config

## Project Structure
```
src/
├── type.py              → Standalone keystroke capture tool
├── graph.py             → Typing pattern visualization
├── randomForest.py      → RF training + evaluation + tree viz
├── data/
│   ├── data.py          → Synthetic "others" data generator
│   ├── ai.py            → Synthetic AI brute-force data generator
│   └── run.py           → Terminal keystroke capture + prediction
├── interface/
│   ├── run.py           → Tkinter GUI login + prediction
│   └── window.py        → Basic Tkinter login stub
└── old/                 → Earlier versions and experiments
    ├── terminal.py      → Original full pipeline (collect → train → predict)
    ├── interface.py     → GUI version of full pipeline
    ├── ANN.py           → Neural network visualization
    ├── SVM.py           → SVM decision boundary demo
    ├── KNN.py           → K-nearest neighbors demo
    ├── generate.py      → Synthetic data generation
    ├── split.py         → Data visualization from train.csv
    └── test.py          → Tkinter centered text demo
```

## Architecture
Keystroke-dynamics biometric auth system:
1. Capture: pynput records press/release timing of 4-digit PIN entry
2. Features: 4 press intervals + 4 release delays = 8 features per sample
3. Train: RandomForestClassifier on CSV data in `tmp/data/`
4. Predict: Real-time identity verification on new keystrokes

## Key Files
| Entry Point | Purpose |
|-------------|---------|
| `src/interface/run.py` | GUI application with login screen |
| `src/data/run.py` | Terminal demo with AI hacking showcase |
| `src/type.py` | Raw keystroke data collection |
| `src/randomForest.py` | Model training and evaluation |
| `src/old/terminal.py` | Full pipeline (legacy, all-in-one) |

## Build & Run
- **Install deps**: `pip install -r requirements.txt`
- **Run GUI**: `python src/interface/run.py`
- **Run terminal demo**: `python src/data/run.py`
- **Collect data**: `python src/type.py`
- **Train model**: `python src/randomForest.py`
- **View patterns**: `python src/graph.py`
- Data directory: `tmp/data/` (must exist with CSV files, gitignored)

## Conventions
- No tests, no CI, no linting
- Git: single `main` branch, informal commit messages
- Data out of source: `tmp/` directory (gitignored via `*tmp` pattern)
- Paths use forward slashes (cross-platform compatible)
