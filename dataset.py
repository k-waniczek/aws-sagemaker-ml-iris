from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, 
    dataset.target, 
    test_size=0.2, 
    random_state=42,
    stratify=dataset.target
)

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

train_array = np.column_stack([y_train, X_train])
test_array = np.column_stack([y_test, X_test])

np.savetxt(output_dir / "train.csv", train_array, delimiter=",", fmt="%g")
np.savetxt(output_dir / "test.csv", test_array, delimiter=",", fmt="%g")

print("train:", train_array.shape, "test:", test_array.shape)