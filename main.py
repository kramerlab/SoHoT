import os
import argparse
from benchmark.run import set_experiments, set_hyperparameter_model_pool, plot_transparency


DATA_DIR = os.getenv('DATA_DIR', './benchmark/data')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './benchmark/data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--seed", type=int)
    args = parser.parse_args()

    data_name, seed = 'SEA50', 0
    if args.dataset is not None:
        data_name = args.dataset
    if args.seed is not None:
        seed = args.seed

    # set_experiments(data_name=data_name, seed=seed, data_dir=DATA_DIR, output_path=OUTPUT_DIR)

    set_hyperparameter_model_pool(data_name, seed=seed, data_dir=".", output_path="./benchmark/data")

    # plot_transparency(data_name='AGR_a', seed=seed, data_dir=DATA_DIR, n_instance_limit=10000)
