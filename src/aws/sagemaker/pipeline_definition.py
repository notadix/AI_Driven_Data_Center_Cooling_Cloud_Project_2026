"""
AWS SageMaker Pipelines definition:
  ProcessingStep → TrainingStep (FNO) → TrainingStep (Safe-PPO) → EvaluationStep → ModelRegistryStep
"""

import os
import boto3

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole")
S3_BUCKET = os.getenv("S3_BUCKET", "datacenter-cooling-mlops")
ECR_IMAGE = os.getenv("ECR_IMAGE", "YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/datacenter-cooling:latest")

PIPELINE_NAME = "DataCenterCoolingMLPipeline"


def get_pipeline():
    try:
        import sagemaker
        from sagemaker.workflow.pipeline import Pipeline
        from sagemaker.workflow.steps import ProcessingStep, TrainingStep
        from sagemaker.workflow.model_step import ModelStep
        from sagemaker.workflow.parameters import ParameterString, ParameterInteger
        from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
        from sagemaker.estimator import Estimator
        from sagemaker.model import Model

        sess = sagemaker.Session(boto3.session.Session(region_name=REGION))

        p_instance = ParameterString(name="ProcessingInstance", default_value="ml.m5.xlarge")
        t_instance = ParameterString(name="TrainingInstance", default_value="ml.g4dn.xlarge")
        fno_epochs = ParameterInteger(name="FNOEpochs", default_value=15)
        rl_episodes = ParameterInteger(name="RLEpisodes", default_value=30)

        processor = ScriptProcessor(
            image_uri=ECR_IMAGE,
            command=["python3"],
            instance_type=p_instance,
            instance_count=1,
            role=ROLE_ARN,
            sagemaker_session=sess,
        )

        step_process = ProcessingStep(
            name="PreprocessTelemetry",
            processor=processor,
            code="dataset/preprocess_telemetry.py",
            inputs=[ProcessingInput(source=f"s3://{S3_BUCKET}/raw/", destination="/opt/ml/processing/input")],
            outputs=[ProcessingOutput(output_name="processed", source="/opt/ml/processing/output",
                                      destination=f"s3://{S3_BUCKET}/processed/")],
        )

        fno_estimator = Estimator(
            image_uri=ECR_IMAGE,
            role=ROLE_ARN,
            instance_count=1,
            instance_type=t_instance,
            entry_point="src/ai/surrogate/train_fno.py",
            hyperparameters={"epochs": fno_epochs, "batch_size": 64, "lr": 1e-3},
            output_path=f"s3://{S3_BUCKET}/models/fno/",
            sagemaker_session=sess,
        )
        step_fno = TrainingStep(
            name="TrainFNOSurrogate",
            estimator=fno_estimator,
            inputs={"train": sagemaker.inputs.TrainingInput(f"s3://{S3_BUCKET}/processed/", content_type="application/x-npy")},
            depends_on=[step_process],
        )

        rl_estimator = Estimator(
            image_uri=ECR_IMAGE,
            role=ROLE_ARN,
            instance_count=1,
            instance_type=t_instance,
            entry_point="src/ai/rl/train_rl.py",
            hyperparameters={"episodes": rl_episodes, "steps": 144},
            output_path=f"s3://{S3_BUCKET}/models/rl/",
            sagemaker_session=sess,
        )
        step_rl = TrainingStep(
            name="TrainSafePPO",
            estimator=rl_estimator,
            depends_on=[step_fno],
        )

        step_evaluate = ProcessingStep(
            name="EvaluateFNOSurrogate",
            processor=processor,
            code="src/ai/surrogate/evaluate_fno.py",
            inputs=[
                ProcessingInput(
                    source=step_fno.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    source=f"s3://{S3_BUCKET}/processed/",
                    destination="/opt/ml/processing/test",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{S3_BUCKET}/evaluation/",
                )
            ],
            depends_on=[step_fno, step_rl],
        )

        model = Model(
            image_uri=ECR_IMAGE,
            model_data=step_fno.properties.ModelArtifacts.S3ModelArtifacts,
            entry_point="src/aws/sagemaker/model_handler.py",
            role=ROLE_ARN,
            sagemaker_session=sess,
        )
        step_register = ModelStep(
            name="RegisterCoolingModel",
            step_args=model.register(
                content_types=["application/json"],
                response_types=["application/json"],
                model_package_group_name="DataCenterCoolingModels",
                approval_status="PendingManualApproval",
            ),
            depends_on=[step_evaluate],
        )

        pipeline = Pipeline(
            name=PIPELINE_NAME,
            parameters=[p_instance, t_instance, fno_epochs, rl_episodes],
            steps=[step_process, step_fno, step_rl, step_evaluate, step_register],
            sagemaker_session=sess,
        )
        return pipeline

    except ImportError:
        print("[!] sagemaker SDK not installed — returning pipeline config dict for offline review.")
        return {
            "pipeline": PIPELINE_NAME,
            "steps": [
                "PreprocessTelemetry",
                "TrainFNOSurrogate",
                "TrainSafePPO",
                "EvaluateFNOSurrogate",
                "RegisterCoolingModel",
            ],
            "s3_bucket": S3_BUCKET,
            "ecr_image": ECR_IMAGE,
            "role_arn": ROLE_ARN,
        }


def main():
    pipeline = get_pipeline()
    if hasattr(pipeline, "upsert"):
        print(f"[*] Upserting SageMaker Pipeline: {PIPELINE_NAME}")
        pipeline.upsert(role_arn=ROLE_ARN)
        execution = pipeline.start()
        print(f"[✓] Pipeline started → ARN: {execution.arn}")
    else:
        import json
        print("[*] Pipeline definition (offline mode):")
        print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    main()
