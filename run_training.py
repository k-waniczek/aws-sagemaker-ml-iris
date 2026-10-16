import time
from sagemaker.sklearn.estimator import SKLearn
from dotenv import load_dotenv
import os

load_dotenv()

BUCKET = os.environ["AWS_S3_BUCKET_NAME"]
ROLE_ARN = os.environ["AWS_SAGEMAKER_ROLE_ARN"]

estimator = SKLearn(
    entry_point="train.py",
    framework_version="1.4-2",
    py_version="py3",
    instance_type="ml.m5.large",
    instance_count=1,
    role=ROLE_ARN,
    hyperparameters={"n_estimators": 100, "random_state": 42},
    output_path=f"s3://{BUCKET}/iris/output",
    base_job_name="iris-sklearn",
)

start = time.time()
estimator.fit({"train": f"s3://{BUCKET}/iris/train/"})
print("training job:", estimator.latest_training_job.name)
print("model artifact:", estimator.model_data)
print(f"elapsed: {time.time() - start:.0f} s")