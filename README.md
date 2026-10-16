# Iris classifier on Amazon SageMaker

## 1. Problem and data
- Task: multiclass classification (setosa / versicolor / virginica).
- Data: Iris dataset from `sklearn.datasets.load_iris` (150 samples, 4 features).
- Split: 80% train / 20% test, `random_state=42`, `stratify` on labels (120 / 30).
- Format: CSV without headers, label in the first column.

## 2. Architecture
Local machine -> S3 (`iris/train/`, `iris/test/`) -> SageMaker training job
(`ml.m5.large`, scikit-learn container 1.4-2) -> `model.tar.gz` in S3 ->
local download, evaluation and prediction (no endpoint).

## 3. How the training job is started
- Method: SageMaker Python SDK v2 from local machine.
- Script: `run_training.py` (starts the job), `train.py` (runs inside the job).
- Models trained in one job:
  - `RandomForestClassifier(n_estimators=100, random_state=42)`
  - `LogisticRegression(max_iter=1000, random_state=42)`

## 4. Versions
- Python: 3.12.7
- scikit-learn: 1.4.2
- numpy: 1.26.4
- sagemaker: 2.257.7
- boto3: 1.43.114
- SageMaker container: `framework_version="1.4-2"`, `py_version="py3"`
- Full list: `requirements.txt`

## 5. AWS configuration
- Region: `eu-north-1`
- S3 bucket: private, public access blocked
  - `iris/train/train.csv`, `iris/test/test.csv`, `iris/output/`, `iris/code/`
- IAM execution role: `SageMakerIrisRole`
  - Trusted service: `sagemaker.amazonaws.com`
  - Permissions: `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` limited to the bucket above;
    `logs:*`, `cloudwatch:PutMetricData`, `ecr:*` for training runtime
- Local credentials: AWS CLI profile (`aws configure`), no keys in code or repo.
- Settings in `.env` (see `.env.sample`):
  `AWS_S3_BUCKET_NAME`, `AWS_SAGEMAKER_ROLE_ARN`, `AWS_DEFAULT_REGION`, `AWS_SAGEMAKER_INSTANCE_TYPE`