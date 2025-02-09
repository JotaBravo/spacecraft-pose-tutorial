import argparse
import mlflow

from lib.logging import load_config, log_params_from_config, flatten_config
from train_swisscube import run_experiment

def main(config_file):
    
    config = load_config(config_file)
    config = flatten_config(config)
    
    # Start MLflow run
    with mlflow.start_run():
        log_params_from_config(config)
        run_experiment(config)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "--config", metavar="DIR", help="Path to the YAML configuration file", required=True)
    args = parser.parse_args()
    
    main(args.cfg)
