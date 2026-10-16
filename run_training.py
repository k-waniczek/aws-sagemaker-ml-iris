import os
import time
from pathlib import Path

from dotenv import load_dotenv
from sagemaker.sklearn.estimator import SKLearn

load_dotenv()

BUCKET = os.environ["AWS_S3_BUCKET_NAME"]
ROLE_ARN = os.environ["AWS_SAGEMAKER_ROLE_ARN"]
INSTANCE_TYPE = os.environ["AWS_SAGEMAKER_INSTANCE_TYPE"]

SCRIPT_DIR = Path(__file__).resolve().parent

estimator = SKLearn(
    entry_point=str(SCRIPT_DIR / "train.py"),
    framework_version="1.4-2",
    py_version="py3",
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    role=ROLE_ARN,
    hyperparameters={"n_estimators": 100, "max_iter": 1000, "random_state": 42},
    output_path=f"s3://{BUCKET}/iris/output",
    code_location=f"s3://{BUCKET}/iris/code",
    base_job_name="iris-sklearn",
)

start = time.time()
estimator.fit({"train": f"s3://{BUCKET}/iris/train/"})
print("training job:", estimator.latest_training_job.name)
print("model artifact:", estimator.model_data)
print(f"elapsed: {time.time() - start:.0f} s")