import pandas as pd
import os


def make_latex_table(metric='Auroc'):
    data_names = ['sleep', 'ann_thyroid', 'churn', 'nursery', 'twonorm', 'optdigits', 'texture', 'satimage',
                  'AGR_a', 'AGR_g', 'RBF_f', 'RBF_m', 'SEA50', 'SEA_5E5', 'HYP_f', 'HYP_m']
    model_names = ['SoHoT', 'HT', 'HAT', 'EFDT', 'SGDClassifier']
    seeds = [0, 1, 2, 3, 4]

    table = pd.DataFrame(index=data_names, columns=model_names)
    for model in model_names:
        for data_name in data_names:
            value = 0
            seeds_seen = 0
            for seed in seeds:
                file_path = f'./data/mogon_results/{model}/{data_name}_summary-seed-{seed}.csv'
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    value += df.iloc[0][metric]
                    seeds_seen += 1
            if seeds_seen > 0:
                value /= seeds_seen
                table.at[data_name, model] = value
    table.index = table.index.str.replace("_", " ")
    print(table.style.highlight_max(axis=1, props="textbf:--rwrap;").to_latex())


if __name__ == '__main__':
    make_latex_table()
