# Methodology

This document complements `README_QUICKSTART.md` with a concise but detailed overview of the project design, modeling choices, forecasting workflow, rebalancing logic, and orchestration setup.

## 1. Project Goal

The goal is to forecast future bike demand at the station level and turn those forecasts into operational outputs.

The current scope includes:
- network-level and station-level forecast visualization
- forecast export through an API
- rebalancing support to help reduce stock shortages

The project is built as an end-to-end prototype covering:
- data preparation
- feature engineering
- model training
- recursive forecasting
- API exposure
- Streamlit visualization
- Airflow orchestration

Important operational note:
- predictions are only possible once the offline dataset has been built and the model artifacts have been trained
- on a fresh setup, initialization must happen first, either through Airflow or by running the build and training scripts manually

## 2. Data Source

The main historical data comes from Capital Bikeshare:
- public trip data for 2024 and 2025
- Washington, DC and its surrounding metro area
- about 856 stations in the current project version

Public source:
- `https://capitalbikeshare.com/system-data`

## 3. Prediction Unit

### Initial and final granularity

The first version of the project was designed at an hourly level.
The current version works with **3 time segments per day**:
- `00h-06h`
- `06h-16h`
- `16h-00h`

The predicted target is `y_station`, defined as the **number of trips** observed for one station during one given segment. In other words, one model observation is:
- `1 station x 1 time segment`

### Why the project moved away from hourly data

Moving from hourly granularity to 3 segments per day helped:
- reduce noise on low-traffic stations
- reduce sparsity in the dataset
- stabilize the prediction target
- keep the output aligned with operational planning needs

This was a trade-off between temporal precision and signal robustness.

## 4. Target Construction

The raw data contains individual trips, not a ready-made forecasting target.

The preparation pipeline therefore:
- reads merged trip files
- assigns each trip to one of the 3 daily segments
- aggregates trip counts by `start_station_id`, `date`, and `segment_id`

The project then builds a **complete station x day x segment panel** covering all stations, all days, and all segments.

Missing combinations are filled with `0`, which helps:
- make no-demand segments explicit
- keep a regular temporal structure
- compute autoregressive features consistently

## 5. Feature Engineering

The final training table, `features_3segments.csv`, includes:

### Calendar features

- `year`
- `month`
- `dayofweek`
- `dayofyear`
- `hour`
- `segment_id`
- `is_weekend`
- `is_holiday`, based on the US federal holiday calendar

### Cyclical encodings

To represent recurring temporal patterns, the project adds sine and cosine encodings for:
- day of week
- day of year
- month
- hour

This is useful because temporal variables are cyclical by nature. For example, `23:00` and `00:00` are far apart numerically but close in time. Using sine and cosine transformations helps the model capture that continuity more naturally.

### Lag features

The current lag set is:
- `lag_1`
- `lag_2`
- `lag_3`
- `lag_21`
- `lag_42`

With 3 segments per day:
- `lag_1` = previous segment
- `lag_3` = roughly 1 day earlier
- `lag_21` = roughly 7 days earlier
- `lag_42` = roughly 14 days earlier

### Rolling features

Rolling statistics are computed independently for each station:
- `roll_mean_3`, `roll_std_3`
- `roll_mean_21`, `roll_std_21`
- `roll_mean_42`, `roll_std_42`

These features are built strictly from past observations to avoid data leakage.

## 6. Modeling Strategy

### 6.1 General idea

Station behavior is highly heterogeneous:
- some stations account for a large share of total demand
- others are noisier and have much lower traffic

A single global model would force a difficult compromise between:
- strong performance on high-volume stations
- stable behavior on low-volume stations

### 6.2 Dual routing model

To address this, the project uses a **dual-model setup**:
- one model for high-volume stations
- one model for low-volume stations

The split is based on cumulative demand volume:
- stations are ranked by total demand
- the `high-volume` group covers stations up to `80%` of cumulative demand
- the remaining stations are assigned to the `low-volume` group

At inference time:
- known high-volume stations go to `model_high`
- known low-volume stations go to `model_low`
- unseen stations fall back to the low-volume model

### 6.3 Learning algorithm

The current model is a `CatBoostRegressor` with:
- `loss_function="Poisson"`
- native handling of categorical variables such as `start_station_id`, `segment_id`, `is_holiday`, and `is_weekend`

CatBoost is a good fit here because it performs well on tabular data, supports categorical variables natively, captures non-linear behavior, and works naturally with count-like targets through a Poisson loss.

### 6.4 Time-aware validation

Cross-validation respects chronology:
- folds are built on dates
- `n_splits=2`
- `test_size=3 * 30`, or 90 segments
- `gap=42` segments between train and validation

The main selection metric is `MAE`.
Additional tracked metrics are:
- `sMAPE`
- `Bias`

### 6.5 Final retraining

Once the structure is validated:
- the full dataset is reloaded
- stations are split again into `high-volume` and `low-volume`
- two final models are trained
- artifacts are saved in `models/`

This step is not optional for first use:
- if no trained artifacts exist yet, the prediction layer cannot run
- the API and Streamlit prediction flow both depend on those saved model files

## 7. Recursive Forecasting

The project produces a **multi-day recursive forecast** rather than a one-shot forecast:
- the forecast start is aligned to the next valid segment boundary
- future slots are generated for `days * 3`
- lag and rolling features are rebuilt from the available history
- each prediction is fed back into the history before predicting the next step

This forecasting stage assumes that:
- the processed feature table already exists
- the trained model artifacts already exist

This makes future forecasting more realistic, but it also means that errors can propagate as the horizon gets farther away.

## 8. Evaluation

The project uses several complementary metrics.

### MAE

`MAE` measures the average absolute error per observation.
Here, one observation means:
- `1 station x 1 segment`

### Bias

`Bias` measures the average signed error:
- positive = overprediction
- negative = underprediction

### WAPE

`WAPE` is not read per observation.
It is a global ratio comparing total absolute error to total observed demand volume.

### sMAPE

`sMAPE` measures relative symmetric error.
It is useful for relative comparison, although it becomes less intuitive on series with many zeros or very low counts.

### 8.1 Recursive backtest results

The recursive backtest in `10_recursive_backtest_days.ipynb` gives the following `J+1` results:
- `MAE = 1.87`
- `Bias = +0.52`
- `WAPE = 29.8%`

Interpretation:
- `MAE = 1.87` means the average error is about `1.87 trips` per `station x segment`
- `Bias = +0.52` means the model still slightly overpredicts demand on average
- `WAPE = 29.8%` means that, at the global `J+1` level, cumulative forecast error is about `30%` of total observed demand volume

Important note:
- `J+1` means the **first forecast day**
- that corresponds to the **first 3 segments** of the horizon
- over `856 stations`, this represents `2568` observations

Performance then degrades gradually as the recursive horizon moves further out, which is expected in this setup.

## 9. Rebalancing

### 9.1 Goal

The rebalancing module turns forecasts into operational suggestions:
- identify stations under pressure
- identify stations that can give bikes
- propose bike transfers
- generate an ordered route plan

### 9.2 Inputs

The logic relies on:
- a network forecast file
- a station capacity file
- a realtime station status file
- a station coordinates file

### 9.3 Assumptions and limitations

The forecasting layer uses real trip data.
However, some rebalancing inputs rely on working assumptions so the prototype can go further operationally.

For example:
- station capacities are stored in `station_capacity.csv`
- the net-out ratio is treated as a heuristic parameter

In the current version:
- `capacity_bikes` is set between `10` and `43`
- `min_buffer` is set between `2` and `9`
- `target_fill_ratio = 0.55`
- `max_fill_ratio = 0.90`
- `net_out_ratio = 0.35` by default in the API

The net-out ratio approximates the share of predicted rentals that actually reduces station stock, in other words, bikes that leave without being offset by arrivals from other trips during the same horizon.

### 9.4 Alert rules

Operationally:
- `pred_out_horizon` = predicted demand over the selected horizon
- `pred_net_out_horizon` = predicted demand adjusted by `net_out_ratio`
- `stock_proj` = projected remaining stock after net consumption

A station is classified as:
- `warning` when a projected deficit appears
- `critical` when projected stock becomes negative or when the deficit reaches `max(5 bikes, 30% of station capacity)`

### 9.5 Transfer strategy

The transfer plan uses a greedy heuristic:
- `critical` receiver stations are handled before `warning` stations
- for each receiver, the algorithm searches for a donor with transferable stock
- when coordinates are available, the chosen donor is the closest one by **straight-line distance**
- otherwise, the donor with the largest available stock is selected

This is not a globally optimal solver. It is a simple, interpretable, and fast heuristic for a practical prototype.

## 10. Airflow Orchestration

The project includes a weekly Airflow retraining DAG:
- `bike_demand_weekly_retrain`

Its role is to:
- rebuild the offline dataset
- retrain the final model

Current schedule:
- every Monday at `02:00`
- timezone `Europe/Paris`

If new historical files are added under `data/raw/2024` or `data/raw/2025`, the DAG can:
- extract zip archives
- rebuild the merged CSV
- regenerate features
- retrain the model

This already moves the project beyond a notebook-only workflow and closer to an industrial setup.

## 11. Technical Stack

- Python
- Pandas
- CatBoost
- FastAPI
- Streamlit
- Airflow
- Docker / Docker Compose

## 12. Current Limitations

The main current limitations are:
- some operational assumptions are still heuristic
- the rebalancing module is not yet connected to a fully reliable realtime system
- observability and monitoring are not fully implemented yet
- production deployment is not yet industrialized

## 13. Next Steps

- add MLflow for experiment and model tracking
- add Grafana and Prometheus for monitoring
- strengthen data quality checks and drift tracking
- industrialize deployment on a VPS with Nginx as a reverse proxy
