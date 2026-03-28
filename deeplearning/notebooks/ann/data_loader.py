import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_fashion_mnist():
    file_path = "fashion-mnist_train.csv"

    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "zalando-research/fashionmnist",
    file_path,
    pandas_kwargs={
        "encoding": "latin1",
        "on_bad_lines": "skip"   # skip corrupted rows
    }
)
    
    return df