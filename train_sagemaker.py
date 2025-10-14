from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
import boto3
import sagemaker

# ===== Global Config =====
REGION = "eu-north-1"
BUCKET = "video-sentimental-analysis"
ROLE = "arn:aws:iam::613890646483:role/sentiment-analysis-excecution-role"

# S3 Paths
S3_TENSORBOARD = f"s3://{BUCKET}/tensorboard"
S3_OUTPUT = f"s3://{BUCKET}/artifacts/"
S3_TRAIN = f"s3://{BUCKET}/dataset/train/"
S3_DEV = f"s3://{BUCKET}/dataset/dev/"
S3_TEST = f"s3://{BUCKET}/dataset/test/"


def start_training():
# Pin session to eu-north-1
        sess = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

        tensorboard_config = TensorBoardOutputConfig(

                s3_output_path=S3_TENSORBOARD,
                container_local_output_path="/opt/ml/output/tensorboard"
        )

        estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role=ROLE,
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={"batch-size": 32,"epochs": 25},
        tensorboard_config=tensorboard_config,
        sagemaker_session=sess,
        output_path=S3_OUTPUT
        )

# Start training
        estimator.fit({
        "training": S3_TRAIN,
        "validation": S3_DEV,
        "test": S3_TEST
    })


if __name__ == "__main__":
    start_training()
