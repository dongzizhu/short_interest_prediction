# Short Interest Forecasting (JP Morgan Project)

This repository develops a full pipeline to forecast biweekly short interest for U.S. securities and to implement and backtest trading strategies driven by those forecasts. Data originates from FINRA’s Consolidated Short Interest files (biweekly), and is augmented with market data (prices/volumes), shares outstanding/float, and corporate actions to ensure clean, bias-free research.

## Objectives

### Forecasting: 
 - Build models that predict next-period short interest (level or change) and derived metrics (e.g., days-to-cover, short-interest-as-%-of-float).
### Signals: 
 - Convert forecasts into tradable signals with clear, deterministic rules.
### Backtesting: 
 - Evaluate strategies with an event-driven backtester that respects data availability (publication lag), transaction costs, and capacity.
### Reproducibility: 
 - Provide a modular codebase, configuration-driven runs, tests, and versioned data transforms.