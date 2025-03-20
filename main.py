import os
import argparse
import sys
from benchmark.run import set_experiments, set_hyperparameter_model_pool, set_ensemble


DATA_DIR = os.getenv('DATA_DIR', './benchmark/data')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './benchmark/data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-e", "--ensemble", type=int)
    parser.add_argument("-s", "--seed", type=int)
    args = parser.parse_args()

    data_name, seed = 'SEA50', 0
    ensemble_testing = False
    if args.dataset is not None:
        data_name = args.dataset
    if args.ensemble is not None:
        ensemble_testing = args.ensemble
        ensemble_size = 10 if ensemble_testing == 1 else 30
    if args.seed is not None:
        seed = args.seed

    # set_experiments(data_name=data_name, seed=seed, data_dir=DATA_DIR, output_path=OUTPUT_DIR)
    # sys.exit()

    if not ensemble_testing:
        set_hyperparameter_model_pool(data_name, seed=seed, data_dir=DATA_DIR, output_path=OUTPUT_DIR)
    else:
        set_ensemble(data_name, ensemble_size=ensemble_size, seed=seed, data_dir=DATA_DIR, output_path=OUTPUT_DIR)
