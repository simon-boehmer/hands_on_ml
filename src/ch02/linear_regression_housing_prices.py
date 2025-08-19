from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from zlib import crc32

# Resolve repo paths
repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data" / "sample"
csv_path = data_dir / "housing.csv"


# Split option 1: sklearn utility
def split_train_test(
    df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_ratio, random_state=random_state)


# Split option 2: manual random shuffle
def split_train_test_manual(
    df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    shuffled = np.random.permutation(len(df))
    test_size = int(len(df) * test_ratio)
    test_indices = shuffled[:test_size]
    train_indices = shuffled[test_size:]
    return df.iloc[train_indices], df.iloc[test_indices]


# Split option 3: stable split by ID (avoids reshuffling issues)
def is_in_test_set(identifier, test_ratio: float, hash=crc32) -> bool:
    return hash(np.int64(identifier)) & 0xFFFFFFFF < test_ratio * 2**32


def split_train_test_by_id(
    df: pd.DataFrame, test_ratio: float, id_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = df[id_column]
    in_test_set = ids.apply(lambda id_: is_in_test_set(id_, test_ratio))
    return df.loc[~in_test_set], df.loc[in_test_set]


def main() -> None:
    # Load dataset
    housing = pd.read_csv(csv_path)

    # Inspect dataset
    print(housing.head())
    print(housing.info())
    print(housing.describe())

    housing.hist(bins=50, figsize=(20, 20))
    plt.show()

    # Alternative splits
    # train2, test2 = split_train_test_manual(housing)
    # housing_with_id = housing.reset_index()  # index as stable ID
    # train3, test3 = split_train_test_by_id(housing_with_id, 0.2, "index")

    # Use sklearn split (default)
    train, test = split_train_test(housing)
    print(f"train: {len(train)}, test: {len(test)}")


if __name__ == "__main__":
    main()
