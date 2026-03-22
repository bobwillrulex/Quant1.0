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

## Discord Continuous Run-All Alerts

In the web UI, use **Continuous Run All + Discord** to:

- Save a Discord webhook URL.
- Start a background monitor that checks all presets included in **Run All** every 10 minutes.
- Only poll during U.S. market hours (Mon–Fri, 9:30 AM–4:00 PM ET).
- Send a Discord message only when a model action changes (e.g., HOLD ➜ BUY).
