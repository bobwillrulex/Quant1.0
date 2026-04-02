# Quant1.0

Refactored quant strategy trainer/presenter with modular code and persistent local storage.

## Project Structure

- `main.py` — Flask routes + CLI entrypoint.
- `quant/ml.py` — model training, strategy evaluation orchestration, and prediction logic.
- `quant/ml_monte_carlo.py` — Monte Carlo simulation/statistics helpers for backtest robustness analysis.
- `docs/ML_FEATURES_AND_OUTPUTS.md` — reference for feature sets and model output payloads.
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

Run Flask through ngrok (shareable public URL):

```bash
export NGROK_AUTHTOKEN="your_token_here"
python main.py --ui --host 127.0.0.1 --port 5000 --ngrok
```

You can also pass the token inline with `--ngrok-authtoken`, but using `NGROK_AUTHTOKEN` is safer because it keeps secrets out of shell history.

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
