import os
import sys
import tarfile
from pathlib import Path

import boto3
import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix

load_dotenv()

BUCKET = os.environ["AWS_S3_BUCKET_NAME"]
OUTPUT_PREFIX = "iris/output"
TEST_KEY = "iris/test/test.csv"


def download_from_s3(key, destination):
    boto3.client("s3").download_file(BUCKET, key, str(destination))


if len(sys.argv) != 2:
    sys.exit("usage: python evaluate.py <job_name>")

job_name = sys.argv[1]
model_key = f"{OUTPUT_PREFIX}/{job_name}/output/model.tar.gz"

work_dir = Path("artifact")
work_dir.mkdir(exist_ok=True)

archive_path = work_dir / "model.tar.gz"
test_path = work_dir / "test.csv"

download_from_s3(model_key, archive_path)
download_from_s3(TEST_KEY, test_path)

with tarfile.open(archive_path) as archive:
    archive.extractall(work_dir)

model_rfc = joblib.load(work_dir / "model_rfc.joblib")
model_lr = joblib.load(work_dir / "model_lr.joblib")

test_data = np.loadtxt(test_path, delimiter=",")
y_test = test_data[:, 0].astype(int)
X_test = test_data[:, 1:]

sample = np.array([[5.1, 3.5, 1.4, 0.2]])

for name, model in [("Random Forest", model_rfc), ("Logistic Regression", model_lr)]:
    y_pred = model.predict(X_test)
    print(f"{name} accuracy:", accuracy_score(y_test, y_pred))
    print(f"{name} confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"{name} sample prediction (0=setosa 1=versicolor 2=virginica):", model.predict(sample)[0])