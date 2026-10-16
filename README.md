# Iris classifier on Amazon SageMaker

## 1. Problem and data
  - Task: multiclass classification (setosa / versicolor / virginica).
  - Data: Iris dataset from `sklearn.datasets.load_iris` (150 samples, 4 features).
  - Split: 80% train / 20% test, `random_state=42`, `stratify` on labels (120 / 30).
  - Format: CSV without headers, label in the first column.

## 2. Architecture
Local machine -> S3 (`iris/train/`, `iris/test/`) -> SageMaker training job (`ml.m5.large`, scikit-learn container) -> `model.tar.gz` in S3 -> local download, evaluation and prediction (no endpoint).

## 3. How the training job is started
  - Method: SageMaker Python SDK (v2) from local machine.
  - Script: `run_training.py` (starts the job), `train.py` (runs inside the job)
  - Model: `RandomForestClassifier(n_estimators=100, random_state=42)`

# 4. Versions:
  - Python: `3.12.x`, scikit-learn `1.4.2`, numpy `<2`, sagemaker `<3`, boto3 `1.43.114`
  - SageMaker container: `framework_version="1.4.2"`, `py_version="py3"`
  - Full list: `requirements.txt`

## 5. AWS configuration
  - Region: `eu-north-1`
  - S3 bucket: (private, public access blocked)
    - `iris/train/train.csv`, `iris/test/test.csv`, `iris/output/`
  - IAM execution role: `SageMakerIrisRole` (trusted service: sagemaker.amazonaws.com),
  permissions limited to the bucket above (s3:GetObject, s3:PutObject, s3:ListBucket)
  - Local credentials: AWS CLI profile (`aws configure`), no keys in code or repo.
  - Settings in `.env` (see `.env.sample`): `AWS_S3_BUCKET_NAME`, `AWS_SAGEMAKER_ROLE_ARN`