# Maya v2: buy-ahead decision tool

End-of-semester version of Maya. **Separate from the original app** (`maya-forecasting` repo,
`maya-streamlit-app` folder), which is unchanged for the client meeting.

Question it answers: *how much power should Maui Hiva buy the day before?* Buying short costs far more than
a small surplus, so Maya v2 forecasts a calibrated range and recommends a buy level, not just a best guess.

## Run it
- Dashboard: double-click `Start Maya v2.bat` (or `streamlit run app/streamlit_app.py`)
- Today's forecast: double-click `Run Daily Forecast.bat` (or `python pipeline/live_forecast.py`).
  Scheduled 14:20, 15:00 and 15:40 UTC; the last run before the 16:00 UTC deadline is locked as the forecast of record.
- Install once: `pip install -r pipeline/requirements.txt -r app/requirements.txt`

## What is where
| Path | What |
|---|---|
| `pipeline/quantile_models.py` | Trains quantile GBM + quantile LSTM (demand) and quantile GBM (renewable). `MAYA_MIN_LAG=48` (default, honest buy-time inputs) or `24` (v1 setting) |
| `pipeline/calibrate_and_score.py` | Daily adaptive calibration, buy-ahead scoring vs PACE and PACE + cushion, cost curve |
| `pipeline/risk_score.py` | Green / Yellow / Red daily risk tier, walk-forward validation, buy-rule test |
| `pipeline/live_forecast.py` | Daily live run (EIA + Open-Meteo), backfills from Sep 2 2026, writes `live/`. `--verify` checks it reproduces the backtest exactly |
| `pipeline/maya_common.py` | Shared calibration and solar physics |
| `pipeline/strategies.py` | Hourly spike-risk model, bid ladder (recommended buy), baselines, all scored at real CAISO prices; also scores the next buy day |
| `pipeline/data/pace_rt_history_2024_2026.csv` | PACE real-time prices Mar 2024 - Apr 2026 (before EDAM), used to train the spike model |
| `pipeline/data/weather_live_*.csv` | Cached ECMWF weather since the training data ends, so each daily run only downloads a few days |
| `pipeline/retrain.py` | Monthly champion / challenger retrain (promotes only if better on the last 45 unseen days); `.github/workflows/monthly_retrain.yml` |
| `pipeline/week_ahead.py` | Week-ahead outlook: daily net-load ranges 2-7 days out, calibrated per lead time; trained monthly |
| `pipeline/data/ecmwf_leadtime_2024_2026.csv.gz` | ECMWF forecasts at 2-7 day lead times since Feb 2024 (Open-Meteo previous runs), for the outlook's honest test |
| `pipeline/make_summary.py` | One-page client summary PDF (`app/data/maya_v2_summary.pdf`), rebuilt daily |
| `live/forecast_archive.parquet` | Every live day's forecast as issued; past days are never recomputed |
| `pipeline/prices.py` | CAISO EDAM day-ahead and real-time PACE prices (OASIS, free); called by the daily run |
| `pipeline/data/pace_prices_2026.csv` | Hourly PACE prices since May 1 2026 |
| `pipeline/out/lag48`, `lag24` | Backtest results for both lag settings |
| `pipeline/models/lag48` | Production models used by the live run |
| `live/` | Latest forecast, live track record, daily drivers |
| `app/` | Streamlit dashboard |
| `.github/workflows/daily_forecast.yml` | Daily GitHub Actions runs at 14:20, 15:00 and 15:40 UTC, before the 16:00 UTC deadline. Commits `live/` and the price file. Optional secret `EIA_API_KEY` |

Real-price settlement (track record and Maya vs PACE tab): buy B day-ahead at the day-ahead price, cover or sell the gap
at the real-time price. Extra cost vs perfect foresight = (real-time - day-ahead) x (actual - buy).

`pipeline/data/` needs `master_hourly_cleaned_ALL_YEARS_v2.csv` and `ecmwf_dayahead_forecast_2024-2026.csv`
(copies of the files in `forecast-dashboard-v2/pipeline`).

## Sizing
The dashboard sidebar scales every MWh and dollar figure to Maui Hiva's average load (placeholder 100 MW), assuming its hourly
pattern follows PACE net load.

## Rebuild everything
```
cd pipeline
set MAYA_MIN_LAG=48
python quantile_models.py && python calibrate_and_score.py && python risk_score.py && python prep_app_data.py
python strategies.py
python live_forecast.py --verify && python live_forecast.py
```
