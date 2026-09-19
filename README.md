# 💱 Currency Fair Value Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![Finance](https://img.shields.io/badge/Finance-Macroeconomics-green.svg)

A quantitative macroeconomic dashboard that visualizes the deviation of daily Forex spot rates from long-term equilibrium models. The application calculates the relative Purchasing Power Parity (PPP) for major currency pairs to determine if a currency is fundamentally overvalued or undervalued.

## 📖 The Macroeconomics: Relative PPP
Purchasing Power Parity (PPP) is a long-term macroeconomic equilibrium model. The relative PPP theory states that the exchange rate between two currencies should naturally adjust to reflect the inflation differential between the two countries. 

If the quote country experiences higher inflation than the base country, the quote currency should depreciate to maintain parity in real purchasing power.

This dashboard calculates the fair value spread using the formula:
**S* = S₀ × (CPI_quote_t / CPI_quote_₀) / (CPI_base_t / CPI_base_₀)**

*Note: PPP is a long-run anchor. Misalignments can last for years and are not standalone trading signals.*

## ✨ Features
* **Automated Data Ingestion:** Fetches decades of daily spot FX data via `yfinance` and official monthly Consumer Price Index (CPI) prints via the `FRED API`.
* **Algorithmic Time-Series Alignment:** Resolves the frequency mismatch between daily market data and monthly macroeconomic data using forward-filling and calendar-time interpolation.
* **Stale Data Extrapolation:** Automatically estimates unpublished CPI figures using the country's trailing 12-month compound inflation rate to prevent the fair-value line from artificially flatlining.
* **Interactive UI:** Built with Streamlit and Plotly for dynamic, responsive financial charting.

## 🛠️ Tech Stack
* **Data Processing:** Pandas, NumPy
* **APIs:** yfinance, FRED (Federal Reserve Economic Data)
* **Frontend:** Streamlit
* **Visualization:** Plotly Graph Objects

## 🚀 Local Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Poet-Philosopher/currency-fair-value-dashboard.git](https://github.com/Poet-Philosopher/currency-fair-value-dashboard.git)
   cd currency-fair-value-dashboard
