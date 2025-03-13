#  NASA Turbofan Engine Degradation - EDA & Preprocessing
This project is focused on interactive exploration and preprocessing of the NASA C-MAPSS dataset, which simulates engine degradation over time. The dataset is used extensively in predictive maintenance, prognostics, 🚀 NASA Turbofan Engine Degradation - EDA & Preprocessing Dashboard

This repository provides a complete **Exploratory Data Analysis (EDA)** and **data preprocessing pipeline** for the NASA C-MAPSS dataset using interactive **Streamlit dashboards** and **Jupyter notebooks**.
The goal is to analyze engine degradation patterns and prepare the data for machine learning tasks like **Remaining Useful Life (RUL)** prediction.

---

## 📌 Project Objectives

- Explore NASA’s C-MAPSS turbofan engine dataset (FD001–FD004)
- Perform comprehensive EDA using Streamlit and Jupyter
- Calculate Remaining Useful Life (RUL) for engines
- Visualize distributions, correlations, outliers, and trends
- Drop redundant features and prepare clean datasets
- Preprocess and scale features for machine learning

---

## 🧩 Dataset Description

The dataset contains multivariate time-series sensor readings from aircraft engines. Each engine has a sequence of operational cycles until failure.

Each file includes:
- **Unit Number** (Engine ID)
- **Time in Cycles**
- **3 Operational Settings**
- **21 Sensor Measurements**
- **RUL Labels** for the test set

Four subsets are provided:
- FD001: 1 Operating condition, 1 Fault mode
- FD002: 6 Operating conditions, 1 Fault mode
- FD003: 1 Operating condition, 2 Fault modes
- FD004: 6 Operating conditions, 2 Fault modes

> 📂 Download dataset from: [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

---

## 📊 Key Features

### 🔍 Exploratory Data Analysis (EDA)
- Dataset selector (FD001–FD004)
- Summary stats: count, mean, std, min, max
- Data type inspection & unique value count
- Visualizations:
  - Engine life bar plots
  - Sensor histograms & boxplots
  - Correlation heatmaps

### ⚙️ RUL Calculation
- Compute Remaining Useful Life for each unit
- Add RUL column dynamically for train/test

### 🧹 Feature Selection
- Drop:
  - Constant features
  - Weakly correlated features
  - Low importance sensors
- Set custom correlation threshold

### 📐 Preprocessing
- Train-test split (x_train, y_train, x_test, y_test)
- Missing value handling
- Feature scaling:
  - StandardScaler
  - MinMaxScaler
  - RobustScaler
  - MaxAbsScaler
- Visualize scaled data distributions

---
## 🙌 Acknowledgments

- 📊 [NASA Prognostics Center of Excellence – C-MAPSS Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
- 📚 [Streamlit – The fastest way to build data apps](https://streamlit.io/)
- 🐼 [Pandas – Python Data Analysis Library](https://pandas.pydata.org/)
- 🧮 [NumPy – The fundamental package for numerical computing](https://numpy.org/)
- 📉 [Matplotlib – Python 2D plotting library](https://matplotlib.org/)
- 🧪 [Seaborn – Statistical data visualization](https://seaborn.pydata.org/)
- 🤖 [Scikit-learn – Machine Learning in Python](https://scikit-learn.org/stable/)




