# Quant1.0

Refactored quant strategy trainer/presenter with modular code and persistent local storage.

## Project Structure

- `main.py` — Flask routes + CLI entrypoint.
- `quant/ml.py` — feature engineering, model training, evaluation, and prediction logic.
- `quant/data.py` — CSV/synthetic data and Yahoo OHLC-to-feature conversion.
- `quant/storage.py` — model bundle persistence and config database.
- `quant/constants.py` / `quant/types.py` — shared constants/types.

## Persistent Database & Model Storage

Model configs are now stored in a **SQLite DB** at:

- `~/.quant1_data/quant_configs.db`

Saved model files are stored at:

- `~/.quant1_data/saved_models/options/*.json`
- `~/.quant1_data/saved_models/spot/*.json`

Because this data is stored in your user home directory (outside the git repo), a `git pull` on this repo will not clear it.
Your DB and saved models remain until you manually delete them.

## Running

```bash
python main.py --ui
```

CLI mode (synthetic data):

```bash
python main.py
```
