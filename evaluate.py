import sys
import tarfile
from pathlib import Path

import boto3
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def download_from_s3(s3_uri, destination):
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    boto3.client("s3").download_file(bucket, key, str(destination))

model_uri = sys.argv[1]  # s3://.../model.tar.gz
test_uri = sys.argv[2]   # s3://.../iris/test/test.csv

work_dir = Path("artifact")
work_dir.mkdir(exist_ok=True)

archive_path = work_dir / "model.tar.gz"
test_path = work_dir / "test.csv"

download_from_s3(model_uri, archive_path)
download_from_s3(test_uri, test_path)

with tarfile.open(archive_path) as archive:
    archive.extractall(work_dir)

model = joblib.load(work_dir / "model.joblib")

test_data = np.loadtxt(test_path, delimiter=",")
y_test = test_data[:, 0].astype(int)
X_test = test_data[:, 1:]

y_pred = model.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred))
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
print("sample prediction (0=setosa, 1=versicolor, 2=virginica):", model.predict(sample)[0])