import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Functions to read the Titanic dataset from CSV files
def load_train_data(TRAIN_PATH: str) -> pd.DataFrame:
    """
    Load the training data from the CSV file.
    """
    return pd.read_csv(TRAIN_PATH)


def load_test_data(TEST_PATH: str) -> pd.DataFrame:
    """
    Load the test data from the CSV file.
    """
    return pd.read_csv(TEST_PATH)


def load_data(TRAIN_PATH: str, TEST_PATH: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both training and test data.
    """
    return load_train_data(TRAIN_PATH), load_test_data(TEST_PATH)


# Functions to summarise statistics of the data
def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the DataFrame with basic statistics.
    """
    print(f"Data shape: {df.shape})")
    print("\n--- Head ---")
    print(df.head())
    print("\n--- Info ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())


def missing_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise missing data in the DataFrame.
    """
    total = df.isnull().sum()
    percent = df.isnull().sum() / df.isnull().count() * 100
    tt = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt["Types"] = types

    return np.transpose(tt)


def frequent_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the most frequent items in the DataFrame.
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ["Total"]
    items = []
    vals = []
    for col in df.columns:
        try:
            itm = df[col].value_counts().index[0]
            val = df[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except TypeError as ex:  # Specify exceptions to catch
            print(ex)
            items.append(0)
            vals.append(0)
            continue
    tt["Most frequent item"] = items
    tt["Frequence"] = vals
    tt["Percent from total"] = np.round(vals / total * 100, 3)

    return np.transpose(tt)


def unique_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the unique items in the DataFrame.
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ["Total"]
    uniques = []
    for col in df.columns:
        unique = df[col].nunique()
        uniques.append(unique)
    tt["Uniques"] = uniques

    return np.transpose(tt)


# Functions for exploratory data analysis
def combine_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the training and test DataFrames into a single DataFrame.
    """
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    all_df = all_df.reset_index(drop=True)
    return all_df


def plot_count_pair(df: pd.DataFrame, feature: str, hue: str | None = None) -> None:
    """
    Plot count pairs for the specified feature (and hue).
    """
    _f, ax = plt.subplots(1, 1, figsize=(8, 4))
    if hue is None:
        sns.countplot(x=feature, data=df)
    else:
        color_list = sns.color_palette()[: df[hue].nunique()]
        sns.countplot(x=feature, data=df, hue=hue, palette=color_list)
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Number of passengers / {feature}")
    plt.show()


def plot_distribution_pair(
    df: pd.DataFrame, feature: str, hue: str | None = None
) -> None:
    """
    Plot distribution pairs for the specified feature (and hue).
    """
    color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]

    # Determine the number of bins for the histogram.
    if df[feature].dtype == "str" or df[feature].dtype == "object":
        bins = "auto"
    else:
        bins = np.histogram_bin_edges(df[feature].dropna(), bins="auto")

    # Plot
    _f, ax = plt.subplots(1, 1, figsize=(8, 4))
    if hue is None:
        sns.histplot(x=feature, data=df, ax=ax)
    else:
        for i, h in enumerate(df[hue].unique()):
            g = sns.histplot(
                df.loc[df[hue] == h, feature],
                color=color_list[i],
                ax=ax,
                label=h,
                bins=bins,
            )
        g.legend()
    ax.set_title(f"Number of passengers / {feature}")
    plt.show()
