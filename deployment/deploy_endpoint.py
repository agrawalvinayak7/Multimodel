from sagemaker.pytorch import PyTorchModel
import sagemaker


def deploy_endpoint():
    session = sagemaker.Session()
    role = "arn:aws:iam::613890646483:role/sentiment-analysis-deploy-endpoint-roll"

    model_uri = "s3://video-sentimental-analysis/inference/model.tar.gz"


    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        entry_point="inference.py",
        source_dir=".",
        sagemaker_session=session,
        name="sentiment-analysis-model",
    )

    # Ensure the SDK uploads code to an existing bucket (no bucket creation)
    model.bucket = "video-sentimental-analysis"

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name="sentiment-analysis-endpoint",
    )


if __name__ == "__main__":
    deploy_endpoint()