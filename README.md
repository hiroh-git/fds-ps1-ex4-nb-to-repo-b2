# Refactor the Jupyter Notebook from the Titanic Kaggle competition

## Purpose

Refactor the Jupyter Notebook from the Titanic Kaggle competition into a well-structured repository with proper software engineering practices.

## Modules Overview

The refactored code extracts the logic from the original Jupyter Notebook into three distinct Python modules within the `titanic/` package:

* **`titanic/data.py`**: Handles data loading, exploratory data analysis (EDA) summarizations (e.g., missing values, frequent items), and visualisation utilities (count and distribution plots).
* **`titanic/feature_engineering.py`**: Handles data transformation and feature creation, such as extracting titles from names, calculating family sizes, binning continuous variables, and encoding categorical features.
* **`titanic/model.py`**: Manages the machine learning pipeline, including splitting predictors and targets, training the Random Forest model, making predictions, and formatting the final Kaggle submission file.

## Setup

This project uses [pixi](https://pixi.sh/) for dependency management.

**1. Clone the repository:**

```bash
git clone https://github.com/hiroh-git/fds-ps1-ex4-nb-to-repo-b2.git
cd fds-ps1-ex4-nb-to-repo-b2
```

**2. Install dependencies:**

```bash
pixi install
```

**3. Select the Kernel:**

Open **`notebooks/titanic_analysis.ipynb`** in your preferred Jupyter environment and select the newly created pixi kernel (**`.pixi/envs/default`**) to run the analysis.

## Data

The data comes from [the Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).
