import pandas as pd


def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a family size feature.
    """
    df_copy = df.copy()
    df_copy["Family Size"] = df_copy["SibSp"] + df_copy["Parch"] + 1

    return df_copy


def create_intervals(df: pd.DataFrame, feature: str, bins: list[float]) -> pd.DataFrame:
    """
    Creates intervals feature.
    """
    df_copy = df.copy()
    df_copy[f"{feature} Interval"] = pd.cut(
        df_copy[feature], bins=bins, labels=False, include_lowest=True
    ).fillna(0)

    return df_copy


def create_sex_pclass(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new feature that combines the Sex and Pclass features.
    """
    df_copy = df.copy()
    df_copy["Sex_Pclass"] = df_copy.apply(
        lambda row: row["Sex"][0].upper() + "_C" + str(row["Pclass"]), axis=1
    )

    return df_copy


def parse_names(row: pd.Series) -> pd.Series:
    """
    Parses the name of a passenger into family name, title, given name, and maiden name.
    """
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except (IndexError, AttributeError, KeyError) as ex:
        print(f"Exception: {ex}")


def create_name_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features by parsing the Name feature into family name, title, given
    name, and maiden name.
    """
    df_copy = df.copy()
    df_copy[["Family Name", "Title", "Given Name", "Maiden Name"]] = df_copy.apply(
        lambda row: parse_names(row), axis=1
    )

    return df_copy


def create_family_type(
    df: pd.DataFrame, bins: list[float], labels: list[str]
) -> pd.DataFrame:
    """
    Creates a family type column.
    """
    df_copy = df.copy()
    df_copy["Family Type"] = pd.cut(df_copy["Family Size"], bins=bins, labels=labels)

    return df_copy


def create_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines the Title feature into a smaller set of titles.
    """
    df_copy = df.copy()
    df_copy["Titles"] = df_copy["Title"]
    # unify `Miss`
    df_copy["Titles"] = df_copy["Titles"].replace("Mlle.", "Miss.")
    df_copy["Titles"] = df_copy["Titles"].replace("Ms.", "Miss.")
    # unify `Mrs`
    df_copy["Titles"] = df_copy["Titles"].replace("Mme.", "Mrs.")
    # unify Rare
    df_copy["Titles"] = df_copy["Titles"].replace(
        [
            "Lady.",
            "the Countess.",
            "Capt.",
            "Col.",
            "Don.",
            "Dr.",
            "Major.",
            "Rev.",
            "Sir.",
            "Jonkheer.",
            "Dona.",
        ],
        "Rare",
    )

    return df_copy


def titles_summary(df: pd.DataFrame) -> None:
    """
    Summarise the titles feature.
    """
    df_copy = df.copy()
    df_copy = (
        df_copy[["Titles", "Sex", "Survived"]]
        .groupby(["Titles", "Sex"], as_index=False)
        .mean()
    )
    print(df_copy)


def one_hot_encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    """
    One hot encodes the Sex feature.
    """
    df_copy = df.copy()
    df_copy["Sex"] = df_copy["Sex"].map({"female": 1, "male": 0}).astype(int)

    return df_copy
