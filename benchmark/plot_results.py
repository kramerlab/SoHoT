import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from benchmark.run import plot_transparency
from critdd import Diagram
from scipy import stats


def get_df_summary(data_names, seeds, model_names, metric, evaluation_method):
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
            if seeds_seen > 0:  # Average results
                value /= seeds_seen
                table.at[data_name, model] = value
    table.index = table.index.str.replace("_", " ")
    return table


# Paired t-test
def get_significances(data_names, seeds, model_names, metric, evaluation_method, significance_level=0.05):
    if len(model_names) != 2: Exception('Perform paired t-test only for two methods!')
    model_a, model_b = {data_name: [] for data_name in data_names}, {data_name: [] for data_name in data_names}
    for i, model in enumerate(model_names):
        for data_name in data_names:
            for seed in seeds:
                file_path = f'./data/mogon_results{evaluation_method}/{model}/{data_name}_summary-seed-{seed}.csv'
                df = pd.read_csv(file_path)
                if i == 0:
                    model_a[data_name].append(df.iloc[0][metric])
                else:
                    model_b[data_name].append(df.iloc[0][metric])
    n_significant_better_model_a, n_significant_better_model_b = 0, 0
    for data_name in data_names:
        t_stat, p_value = stats.ttest_rel(model_a[data_name], model_b[data_name])
        if p_value < significance_level:
            if np.mean(model_a[data_name]) > np.mean(model_b[data_name]):
                n_significant_better_model_a += 1
            else:
                n_significant_better_model_b += 1
    print(f"Model A is significant better on {n_significant_better_model_a} datasets and model B on {n_significant_better_model_b}.")
    return n_significant_better_model_a, n_significant_better_model_b


def make_latex_table(data_names, seeds, model_names, metric, evaluation_method):
    table = get_df_summary(data_names=data_names, seeds=seeds, model_names=model_names, metric=metric,
                           evaluation_method=evaluation_method)

    # Add mean ranks
    ranks = table.rank(axis=1, method='average', ascending=False)
    mean_ranks = ranks.mean()
    table.loc['Mean Rank'] = mean_ranks

    print(table.style.highlight_max(axis=1, props="textbf:--rwrap;").format(precision=3).to_latex())


def compare_sohot_ht(data_names, seeds, metric, evaluation_method):
    fontsize = 23
    model_names = ['SoHoT', 'HT', 'HT_limit']
    n_significance_table = pd.DataFrame(index=model_names, columns=model_names)
    for model_a in model_names: # set diagonal to zero
        n_significance_table.at[model_a, model_a] = 0
    for model_a, model_b in [('SoHoT', 'HT'), ('SoHoT', 'HT_limit'), ('HT', 'HT_limit')]:
        m_names = [model_a, model_b]
        n_significant_better_model_a, n_significant_better_model_b = get_significances(data_names, seeds, m_names,
                                                                                       metric, evaluation_method)
        n_significance_table.at[model_a, model_b] = n_significant_better_model_a
        n_significance_table.at[model_b, model_a] = n_significant_better_model_b

    plt.figure(figsize=(9, 7))
    n_significance_table = n_significance_table.astype(float)
    ax = sns.heatmap(n_significance_table, annot=True, linewidths=0.5, cmap="Blues", annot_kws={"size": fontsize})  # cmap="coolwarm"
    plt.xticks(fontsize=fontsize, rotation=45)
    plt.yticks(fontsize=fontsize, rotation=0)
    cbar = ax.collections[0].colorbar  # Get color bar object
    cbar.ax.tick_params(labelsize=fontsize)  # Set font size for scale labels
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"data/images_sohot/paper/figure-heatmap.pdf", format="pdf")


# https://mirkobunse.github.io/critdd/
# Install with: pip install 'critdd @ git+https://github.com/mirkobunse/critdd'
def get_cridd(data_names, seeds, model_names, metric, evaluation_method):
    df = get_df_summary(data_names=data_names, seeds=seeds, model_names=model_names, metric=metric,
                        evaluation_method=evaluation_method)
    df = df.rename_axis('dataset_name', axis=0)
    df = df.rename_axis('classifier_name', axis=1)
    df = df.astype(np.float64)

    # create a CD diagram from the Pandas DataFrame
    diagram = Diagram(
        df.to_numpy(),
        treatment_names=df.columns,
        maximize_outcome=True
    )

    # inspect average ranks and groups of statistically indistinguishable treatments
    diagram.average_ranks  # the average rank of each treatment
    diagram.get_groups(alpha=.05, adjustment="holm")

    # export the diagram to a file
    diagram.to_file(
        f"./data/images_sohot/critdd-{metric}.tex",
        alpha=.05,
        adjustment="holm",
        reverse_x=True,
        # axis_options={"title": ""},
    )


def compare_sohot_hat(data_names, seeds, evaluation_method):
    plot_scatter = False
    model_names = ['SoHoT', 'HAT']
    table_avg = None
    for i in range(12):
        metric = f"Avg Tree {i} Complexity"
        table = get_df_summary(data_names, seeds, model_names, metric, evaluation_method).abs()
        if table_avg is None:
            table_avg = table
        else:
            table_avg = pd.concat([table_avg, table])
    table_avg = table_avg.groupby(level=0).mean()
    # Visualization: Scatter plot?
    if plot_scatter:
        plt.scatter(data_names, table_avg['SoHoT'])
        plt.scatter(data_names, table_avg['HAT'])
        plt.xticks(rotation=45)
        plt.ylabel("Number of nodes")
        plt.legend(model_names)
        plt.tight_layout()
        plt.show()

    # Show trade-off: Predictive performance / Tree size
    table_auc = get_df_summary(data_names, seeds, model_names, 'Auroc', evaluation_method)
    plt.scatter(data_names, table_auc['SoHoT'] / table_avg['SoHoT'])
    plt.scatter(data_names, table_auc['HAT'] / table_avg['HAT'])
    plt.xticks(rotation=45)
    plt.ylabel("Number of nodes")
    plt.legend(model_names)
    plt.tight_layout()
    plt.show()


def get_table_performance_complexity(data_names, seeds, evaluation_method):
    model_names = ['SoHoT', 'HT', 'HAT', 'EFDT', 'SGDClassifier', 'TEL']
    table_avg = None
    for i in range(12):
        metric = f"Avg Tree {i} Complexity"
        table = get_df_summary(data_names, seeds, model_names, metric, evaluation_method).abs()
        if table_avg is None:
            table_avg = table
        else:
            table_avg = pd.concat([table_avg, table])
    table_complexity = table_avg.groupby(level=0).mean()
    table_complexity['TEL'] = [149.3] * len(table_complexity['SoHoT'])      # TEL: #nodes=2*node_depth ((2*4*(2^5 + 2^6 + 2^7))/12)

    table_auc = get_df_summary(data_names, seeds, model_names, 'Auroc', evaluation_method)
    # Apply log_2 to each entry
    table_complexity.applymap(np.log2)
    table_trade_off = table_auc.div(table_complexity)

    for data_name in data_names:
        data_name = data_name.replace("_", " ")
        row = f"{data_name} & AUC"
        idx_max_auc = table_auc.idxmax(axis=1, skipna=True)
        for i, model_name in enumerate(model_names):
            # if model_name == idx_max_auc.loc[data_name]: row += f" & \\textbf{{{table_auc[model_name][data_name]:.3f}}}"
            # else:
            row += f" & {table_auc[model_name][data_name]:.3f}"
        row += "\\\\\n"
        row += f" & Efficiency "
        idx_max_auc = table_trade_off.idxmax(axis=1, skipna=True)
        for model_name in model_names:
            if model_name.__eq__('SGDClassifier'):
                row += f" & - "
            else:
                if model_name == idx_max_auc.loc[data_name]:
                    row += f" & \\textbf{{{table_trade_off[model_name][data_name]:.3f}}}"
                else:
                    row += f" & {table_trade_off[model_name][data_name]:.3f}"
        row += "\\\\"
        print(row)

    # Add mean ranks
    def print_ranks(ranks, metric_name):
        mean_ranks = ranks.mean()
        ranks = f" & Mean rank {metric_name}"
        for mean_rank in mean_ranks:
            ranks += f" & {mean_rank:.3f}"
        ranks += " \\\\"
        print(ranks)
    print_ranks(table_auc.rank(axis=1, method='average', ascending=False), "AUC")
    print_ranks(table_trade_off.rank(axis=1, method='average', ascending=False), "efficiency")

    # print(table_trade_off.idxmax(axis=1, skipna=True))

if __name__ == '__main__':
    metric = 'Auroc'
    evaluation_method = "/hyperparameter_tuned_results"
    data_names = ['sleep', 'ann_thyroid', 'churn', 'nursery', 'twonorm', 'optdigits', 'texture', 'satimage',
                  'AGR_a', 'AGR_g', 'RBF_f', 'RBF_m', 'SEA50', 'SEA_5E5', 'HYP_f', 'HYP_m',
                  'epsilon', 'poker', 'covtype', 'kdd99'
                  ]
    seeds = [0, 1, 2, 3, 4]

    # 1. Evaluation: SoHoT vs. HT and HT_limit
    # compare_sohot_ht(data_names, seeds, metric, evaluation_method)

    # 2. Evaluation: SoHoT vs. state-of-the-art-ensemble methods
    model_names = ['SoHoT', 'HT', 'HAT', 'EFDT', 'SGDClassifier']
    # make_latex_table(data_names, seeds, model_names, metric, evaluation_method)

    # Ensembles
    make_latex_table(data_names, seeds, ['SoHoT', 'ARF', 'SRP'], metric, "/ensemble_10_results")

    # 2.2 Statistical tests
    # get_significances(data_names, seeds, ['SoHoT', 'HAT'], metric, evaluation_method)

    # 3. Evaluation: Critical difference diagram
    model_names = ['SoHoT', 'HT', 'HAT', 'EFDT', 'SGDClassifier']
    # get_cridd(data_names=data_names, seeds=seeds, model_names=model_names, metric=metric,
    #           evaluation_method=evaluation_method)

    # 3.2 SoHoT vs. HAT
    # compare_sohot_hat(data_names, seeds, evaluation_method)
    # get_table_performance_complexity(data_names, seeds, evaluation_method)

    # 4. Evaluation: SoHoT vs. Soft tree

    # 5. Evaluation: Transparency of SoHoTs
    visualize_tree_at = [2400, 2600, 4900, 5100, 7000, 7400, 7600]
    # visualize_tree_at = [i for i in range(1, 10000, 100)]
    # plot_transparency(data_name='AGR_small', seed=1, visualize_tree_at=visualize_tree_at, save_img=True)
