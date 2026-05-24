# OtoMoto Motorcycle Market Analysis

This project is designed to explore the secondary motorcycle market in Poland by scraping data from OtoMoto.pl, cleaning it, and performing various data science analyses.

## 🚀 Features

- **Web Scraping**: Automated data collection using Playwright, supporting pagination (up to 10 pages).
- **Data Cleaning**: 
  - Brand name normalization.
  - Removal of duplicates and incomplete records.
  - Outlier detection using the Interquartile Range (IQR) method.
- **Data Analysis**:
  - Market share analysis for brands and motorcycle types.
  - Price and mileage distribution visualizations.
  - Linear Regression to predict "fair prices" based on year and mileage.
  - K-means Clustering to segment the market (e.g., Collectors vs. Budget).
  - **Bargain Finder**: An algorithm that identifies underpriced listings based on the regression model.

## 🛠️ Technology Stack

- **Language**: Python 3.13+
- **Project Management**: [uv](https://github.com/astral-sh/uv)
- **Scraping**: Playwright
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## 📦 Installation & Setup

### 1. Install `uv`
If you don't have `uv` installed, follow the instructions at [astral.sh/uv](https://astral.sh/uv).

### 2. Initialize Project
```bash
# Install dependencies
uv sync

# Install Playwright browsers
uv run playwright install
```

## 🏃 How to Run

### 1. Configuration
Edit `config/settings.yaml` to set your desired starting URL and analysis parameters:
- `start_url`: The OtoMoto category page you want to scrape.
- `max_pages`: Number of pages to scrape (default: 100).
- `data_source`: Source of data (`scrape`, `raw`, or `processed`).
- `clustering_algorithm`: Algorithm to use (`auto`, `kmeans`, or `dbscan`).
- `debug`: Set to `true` for detailed logs.

### 2. Execute the Pipeline
Run the main orchestrator:
```bash
uv run main.py
```

## 📂 Project Structure

- `config/`: Configuration files.
- `data/`: Raw and processed datasets.
- `src/`: Source code.
  - `scraper/`: Playwright logic for data acquisition.
  - `cleaning/`: Data normalization and filtering.
  - `analysis/`: Mathematical models and statistics.
  - `utils/`: Helpers for logging and config loading.
- `tests/`: Unit and integration tests.

## 📊 Analysis Methodology

### K-means Clustering
The project segments the market using two synthetic features:
- **Depreciation Coefficient**: $\frac{\text{Price}}{\text{Age}}$
- **Exploitation Coefficient**: $\frac{\text{Price}}{\text{Mileage}}$

### Bargain Search Algorithm
A listing is flagged as a bargain if:
$$\text{Actual Price} < \text{Predicted Price} \times (1 - \text{Threshold})$$
where the Predicted Price is derived from a Linear Regression model:
$$\text{Price} = \beta_0 + \beta_1 \cdot \text{Year} + \beta_2 \cdot \text{Mileage}$$
