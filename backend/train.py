import argparse
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
import mlflow
import mlflow.h2o
import json

def parse_args():
    parser = argparse.ArgumentParser(description="H2O AutoML Training and MLflow Tracking")
    parser.add_argument('--name', default='automl-insurance', help='Name of Experiment. Default is automl-insurance', type=str)
    parser.add_argument('--target', required=True, help='Name of Target Column (y)', type=str)
    parser.add_argument('--models', default=10, help='Number of AutoML models to train. Default is 10', type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    h2o.init()
    mlflow.set_experiment(args.name)
    experiment = mlflow.get_experiment_by_name(args.name)
    print(f"Experiment details:\nName: {args.name}\nExperiment_id: {experiment.experiment_id}\nArtifact Location: {experiment.artifact_location}\nLifecycle_stage: {experiment.lifecycle_stage}\nTracking uri: {mlflow.get_tracking_uri()}")
    main_frame = h2o.import_file(path='data/processed/train.csv')
    with open('data/processed/train_col_types.json', 'w') as fp:
        json.dump(main_frame.types, fp)

    target = args.target
    predictors = [n for n in main_frame.col_names if n != target]
    main_frame[target] = main_frame[target].asfactor()
    with mlflow.start_run():
        aml = H2OAutoML(
                        max_models=args.models,
                        seed=42,
                        balance_classes=True,
                        sort_metric='logloss',
                        verbosity='info',
                        exclude_algos = ['GLM', 'DRF'],
                    )
        
        aml.train(x=predictors, y=target, training_frame=main_frame)
        
        mlflow.log_metrics({"log_loss": aml.leader.logloss(), "AUC": aml.leader.auc()})
        
        mlflow.h2o.log_model(aml.leader, artifact_path="model")
        
        model_uri = mlflow.get_artifact_uri("model")
        print(f'AutoML best model saved in {model_uri}')
        
        lb = get_leaderboard(aml, extra_columns='ALL')
        lb_path = f'{model_uri}/leaderboard.csv'
        lb.as_data_frame().to_csv(lb_path, index=False) 
        print(f'AutoML Complete. Leaderboard saved in {lb_path}')

if __name__ == "__main__":
    main()
