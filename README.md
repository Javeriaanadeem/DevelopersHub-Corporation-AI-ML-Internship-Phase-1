# DevelopersHub Corporation — AI/ML Internship Tasks

**Intern:** Javeria

**Department:** AI/ML Engineering

**DHC ID:** DHC-2448

**Date:** 11th May, 2026

---

This repository contains all the work completed during the AI/ML internship at DevelopersHub Corporation. There are four tasks in total, each covering a different area of data science and machine learning — ranging from basic exploratory data analysis all the way to building a conversational health chatbot using a large language model.

---

## Repository Structure

```
├── task1.ipynb        # Exploratory Data Analysis on Iris Dataset
├── task2.ipynb        # Stock Price Prediction using Linear Regression
├── task3.ipynb        # Heart Disease Classification using Logistic Regression
├── task4.ipynb        # General Health Query Chatbot using Mistral-7B
├── heart.csv          # Dataset used in Task 3
└── README.md
```

---

## Task 1 — Exploratory Data Analysis (EDA)

### Objective
Perform a comprehensive exploratory data analysis on a well-known classification dataset to understand its structure, distributions, and feature relationships through statistical summaries and visualizations.

### Dataset Used
**Iris Dataset** — loaded directly via `seaborn.load_dataset('iris')`
- 150 samples, 5 columns
- Features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- Target: `species` (Setosa, Versicolor, Virginica)

### Methods Applied
- Dataset inspection: `.shape`, `.info()`, `.describe()`, `.head()`
- Scatter plots (Sepal Length vs Width by species)
- Feature histograms
- Box plots for outlier detection
- Correlation heatmap
- Pair plot with species-level hue

### Key Findings
- Setosa is clearly linearly separable from the other two species based on petal dimensions.
- Petal length and petal width show the strongest positive correlation among all features.
- Sepal width shows the most overlap across species, making it the least discriminative feature alone.
- Box plots reveal that Virginica generally has the largest feature values, while Setosa has the smallest petals.

---

## Task 2 — Stock Price Prediction

### Objective
Build a machine learning model to predict the next day's closing price of Apple Inc. (AAPL) stock using historical OHLCV data and evaluate its performance.

### Dataset Used
**Apple Inc. (AAPL) Stock Data** — fetched via `yfinance`
- Date range: January 2023 – January 2024
- Features used: `Open`, `High`, `Low`, `Volume`
- Target: Next day's `Close` price (shifted by 1)

### Model Applied
**Linear Regression** (`sklearn.linear_model.LinearRegression`)
- 80/20 chronological train-test split (no shuffling to preserve time-series order)

### Key Results & Findings
- The model successfully learned the general trend of AAPL's closing prices.
- **Evaluation Metrics:**
  - **MAE (Mean Absolute Error):** Low, indicating predictions are close to actual values on average.
  - **RMSE (Root Mean Squared Error):** Slightly higher than MAE, suggesting some larger deviations on volatile days.
- The Actual vs Predicted plot shows that the model tracks price movement well but struggles with sudden spikes or drops.
- Linear Regression provides a strong baseline; more advanced models (e.g., LSTM, XGBoost) could improve performance on volatile periods.

---

## Task 3 — Heart Disease Classification

### Objective
Build a binary classification model to predict the presence or absence of heart disease in patients, and evaluate model performance using accuracy, AUC score, confusion matrix, and ROC curve.

### Dataset Used
**Heart Disease Dataset** (`heart.csv`)
- 303 patient records, 14 columns
- Features include: `age`, `sex`, `cp` (chest pain type), `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
- Target: `target` (1 = Heart Disease Present, 0 = Absent)
- No missing values

### Model Applied
**Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
- Feature scaling using `StandardScaler`
- 80/20 stratified train-test split (`random_state=42`)
- `max_iter=1000` for convergence

### Key Results & Findings
- **Accuracy:** ~85%+
- **AUC Score:** ~0.91 — indicating excellent discriminative ability
- The confusion matrix shows the model performs well on both classes with minimal false negatives, which is critical in medical diagnosis.
- The ROC curve confirms strong model performance well above the random baseline.
- **Feature Importance (Logistic Coefficients):**
  - `cp` (chest pain type) and `thalach` (max heart rate) are the strongest positive predictors of heart disease.
  - `ca` (number of major vessels) and `oldpeak` (ST depression) are strong negative predictors.
- The correlation heatmap revealed meaningful relationships between cardiovascular indicators.

---

## Task 4 — General Health Query Chatbot

### Objective
Build a conversational health chatbot capable of answering general medical queries using a large language model, with a built-in safety filter to handle sensitive or crisis-related inputs responsibly.

### Dataset Used
No dataset — the chatbot uses a **pre-trained LLM** via Hugging Face Inference API.

### Model Applied
**Mistral-7B-Instruct-v0.1** — accessed via Hugging Face Inference API
- Endpoint: `mistralai/Mistral-7B-Instruct-v0.1`
- Prompt formatted using Mistral's `[INST]...[/INST]` instruction template
- Parameters: `max_new_tokens=200`, `temperature=0.7`

### Safety Features
- **Keyword-based unsafe input filter** — detects crisis-related phrases (e.g., self-harm, suicide) and redirects users to professional help immediately.
- System prompt instructs the model to always recommend seeing a doctor for serious issues.

### Key Results & Findings
- The chatbot successfully responds to general health questions in a clear and concise manner.
- The safety filter correctly intercepts sensitive inputs before they reach the model, ensuring responsible deployment.
- Temperature of 0.7 balances creativity and factual accuracy in responses.
- The chatbot appropriately defers complex or serious medical questions to healthcare professionals, reducing risk of harmful advice.
- Limitation: The keyword filter is rule-based and may miss paraphrased unsafe inputs; a more robust approach would use a fine-tuned safety classifier.

---

## Technologies & Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `yfinance` | Stock market data retrieval |
| `requests` | API calls to Hugging Face |

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/dhc-internship-tasks.git
   cd dhc-internship-tasks
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn yfinance requests
   ```

3. For **Task 3**, ensure `heart.csv` is in the same directory as `task3.ipynb`.

4. For **Task 4**, replace `YOUR_HF_TOKEN_HERE` in `task4.ipynb` with your Hugging Face API token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

5. Open any notebook in Jupyter:
   ```bash
   jupyter notebook
   ```

---

## About the Internship

All tasks in this repository were carried out during an AI/ML Engineering internship at **DevelopersHub Corporation**. The internship offer was formally issued on **11th May 2026** and authorized by **M. Faizan Zafar Khan**, General Manager Operations.

**Website** [www.developershubcorp.com](https://www.developershubcorp.com) 

**Contact** [hr@developershubcorp.com](mailto:hr@developershubcorp.com) 
