# Retirement Monte Carlo Simulation

This project runs a Monte Carlo simulation for the retirement planning assignment in Decision Models class at Duke University, Fuqua School of Business

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project structure

```text
retirement-monte-carlo/
├── data/
│   └── retirement_planning_data_2026.xlsx
├── outputs/
├── src/
│   └── retirement_monte_carlo.py
└── requirements.txt
```

## Run

```bash
python src/retirement_monte_carlo.py
```

The script will:
- read the historical monthly returns from the Excel file
- fit distributions to S&P 500 and Treasury returns
- simulate 72 months of investing $2,000 per month
- compare 0%, 25%, 50%, 75%, and 100% S&P 500 allocations
- save a CSV summary and an overlay CDF chart in `outputs/`
