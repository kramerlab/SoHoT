import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from benchmark.run import plot_transparency


def get_df_summary(model_names, metric, evaluation_method):
    data_names = ['sleep', 'ann_thyroid', 'churn', 'nursery', 'twonorm', 'optdigits', 'texture', 'satimage',
                  'AGR_a', 'AGR_g', 'RBF_f', 'RBF_m', 'SEA50', 'SEA_5E5', 'HYP_f', 'HYP_m',
                  'epsilon','poker', 'covtype', 'kdd99'
                  ] # missing:
    seeds = [0, 1, 2, 3, 4]

    table = pd.DataFrame(index=data_names, columns=model_names)
    for model in model_names:
        for data_name in data_names:
            value = 0
            seeds_seen = 0
            for seed in seeds:
                file_path = f'./data/mogon_results{evaluation_method}/{model}/{data_name}_summary-seed-{seed}.csv'
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        value += df.iloc[0][metric]
                        seeds_seen += 1
                    except:
                        pass
            if seeds_seen > 0:
                value /= seeds_seen
                table.at[data_name, model] = value
    table.index = table.index.str.replace("_", " ")
    return table


def make_latex_table(model_names, metric, evaluation_method):
    table = get_df_summary(model_names=model_names, metric=metric, evaluation_method=evaluation_method)

    # Add mean ranks
    ranks = table.rank(axis=1, method='average', ascending=False)
    mean_ranks = ranks.mean()
    table.loc['Mean Rank'] = mean_ranks

    print(table.style.highlight_max(axis=1, props="textbf:--rwrap;").format(precision=3).to_latex())


def compare_sohot_ht(model_names, metric, evaluation_method):
    table = get_df_summary(model_names=model_names, metric=metric, evaluation_method=evaluation_method)
    table = table.apply(pd.to_numeric, errors='coerce')
    # table = table.T

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(table, annot=True, fmt=".3f", linewidths=0.5)   #  cmap="coolwarm"
    plt.savefig(f"data/images_sohot/paper/figure-heatmap.pdf", format="pdf")
    # plt.show()


if __name__ == '__main__':
    metric = 'Auroc'
    evaluation_method = "/hyperparameter_tuned_results"

    # 1. Evaluation: SoHoT vs. HT and HT_limit
    model_names = ['SoHoT', 'HT', 'HT_limit']
    compare_sohot_ht(model_names, metric, evaluation_method)

    # 2. Evaluation: SoHoT vs. state-of-the-art-ensemble methods
    model_names = ['SoHoT', 'HT', 'HAT', 'EFDT', 'SGDClassifier']
    make_latex_table(model_names, metric, evaluation_method)

    # 3. Evaluation: SoHoT vs. Soft tree

    # 4. Evaluation: Transparency of SoHoTs
    visualize_tree_at = [2400, 2600, 4900, 5100, 7000, 7400, 7600]
    # visualize_tree_at = [i for i in range(1, 10000, 100)]
    # plot_transparency(data_name='AGR_small', seed=1, visualize_tree_at=visualize_tree_at, save_img=True)
