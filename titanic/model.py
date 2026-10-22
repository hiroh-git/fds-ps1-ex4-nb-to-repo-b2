import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def split_predictors_target(
    df: pd.DataFrame, predictors: list, target: str | None = None
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Split the dataset into predictors and target variables.
    """
    df_X = df[predictors]
    df_Y = df[target].values if target is not None else None
    return df_X, df_Y


def train_rf(
    train_X: pd.DataFrame,
    train_Y: pd.Series,
    n_estimators: int,
    criterion: str,
    random_state: int,
    n_jobs: int,
) -> RandomForestClassifier:
    """
    Fits a Random Forest Classifier and returns the trained classifier.
    """
    # Initialize model using config values with fallback defaults
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=False,
    )

    # Fit classifier
    clf.fit(train_X, train_Y)

    return clf


def predict_rf(
    clf: RandomForestClassifier,
    df_X: pd.DataFrame,
) -> pd.Series:
    """
    Generates predictions using the trained classifier.
    """
    preds = pd.Series(clf.predict(df_X), index=df_X.index)
    return preds


def train_and_predict_rf(
    train_X: pd.DataFrame,
    train_Y: pd.Series,
    valid_X: pd.DataFrame,
    n_estimators: int,
    criterion: str,
    random_state: int,
    n_jobs: int,
) -> tuple[RandomForestClassifier, pd.Series, pd.Series]:
    """
    Fits a Random Forest Classifier and returns the trained classifier and predictions.

    """
    # Initialize model using config values with fallback defaults
    clf = train_rf(
        train_X=train_X,
        train_Y=train_Y,
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # Generate predictions
    preds_tr = predict_rf(clf, train_X)
    preds_va = predict_rf(clf, valid_X)

    return clf, preds_tr, preds_va


def create_submission_file(
    test_df: pd.DataFrame, preds_te: pd.Series, filename: str
) -> None:
    """
    Creates a submission file for Kaggle competition.
    """
    df = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": preds_te})
    df.to_csv(filename, index=False)
