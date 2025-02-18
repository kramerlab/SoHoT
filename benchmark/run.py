from benchmark.evaluator import TreeEvaluator
from benchmark.load_data import load_data_stream
from benchmark.hyperparameter_tuning_pool import ModelPool
from sohot.tree_visualization import visualize_soft_hoeffding_tree
from sohot.sohot_ensemble_layer import SoftHoeffdingTreeLayer
from capymoa.classifier import HoeffdingTree, EFDT, HoeffdingAdaptiveTree, SGDClassifier, \
    AdaptiveRandomForestClassifier, StreamingRandomPatches
from capymoa.splitcriteria import SplitCriterion, _split_criterion_to_cli_str
from capymoa.stream import Schema
from pathlib import Path
import pandas as pd
from capymoa.instance import Instance
import re
import torch
import yaml
import itertools
from typing import Union


# In CapyMOA Version 0.8 SGDClassifier does not return anything!
class SGDClassifierMod(SGDClassifier):
    def predict_proba(self, instance: Instance):
        if not self._trained_at_least_once: return None
        return self.sklearner.predict_proba([instance.x])[0]


class HoeffdingTreeLimit(HoeffdingTree):
    def __init__(
            self,
            schema: Schema | None = None,
            random_seed: int = 0,
            grace_period: int = 200,
            split_criterion: Union[str, SplitCriterion] = "InfoGainSplitCriterion",
            confidence: float = 1e-3,
            tie_threshold: float = 0.05,
            leaf_prediction: int = "NaiveBayesAdaptive",
            nb_threshold: int = 0,
            numeric_attribute_observer: str = "GaussianNumericAttributeClassObserver",
            binary_split: bool = False,
            max_byte_size: float = 33554433,
            memory_estimate_period: int = 1000000,
            stop_mem_management: bool = True,
            remove_poor_attrs: bool = False,
            disable_prepruning: bool = True,
            node_limit: int | None = None,
    ):
        self.node_limit = node_limit
        super().__init__(schema=schema, random_seed=random_seed, grace_period=grace_period,
                         split_criterion=split_criterion, confidence=confidence,
                         tie_threshold=tie_threshold, leaf_prediction=leaf_prediction,
                         nb_threshold=nb_threshold, numeric_attribute_observer=numeric_attribute_observer,
                         binary_split=binary_split, max_byte_size=max_byte_size,
                         memory_estimate_period=memory_estimate_period, stop_mem_management=stop_mem_management,
                         remove_poor_attrs=remove_poor_attrs, disable_prepruning=disable_prepruning)

    def train(self, instance):
        tree_size = self.moa_learner.getNodeCount()
        if tree_size > self.node_limit:
            self.moa_learner.gracePeriodOption.setValue(10000000)
            # self.moa_learner.growth_limit = False     # protected attribute cannot be reached
        super().train(instance)


def set_experiments(data_name, seed=1, data_dir=".", output_path="./benchmark/data"):
    Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/evaluation_results").mkdir(parents=True, exist_ok=True)
    data_stream, n_instance_limit = load_data_stream(data_name, seed=seed, data_dir=data_dir)
    schema = data_stream.get_schema()
    output_path += "/evaluation_results"
    models = [
        (SoftHoeffdingTreeLayer(schema=schema, trees_num=1, seed=seed), f"{output_path}/SoHoT"),
        (HoeffdingTree(schema=schema, random_seed=seed), f"{output_path}/HT"),
        (HoeffdingTreeLimit(schema=schema, random_seed=seed, node_limit=127), f"{output_path}/HTLimit"),
        (HoeffdingAdaptiveTree(schema=schema, random_seed=seed), f"{output_path}/HAT"),
        (EFDT(schema=schema, random_seed=seed), f"{output_path}/EFDT"),
        (SGDClassifierMod(schema=schema, loss='log_loss', random_seed=seed), f"{output_path}/SGDClassifier")
    ]
    for model, output_path in models:
        compute_complexity = output_path.endswith('SoHoT')
        run_experiments(data_stream=data_stream, data_name=data_name, model=model, n_instance_limit=n_instance_limit,
                        seed=seed, output_path=output_path, compute_complexity=compute_complexity)


def set_hyperparameter_model_pool(data_name, seed=1, data_dir=".", output_path="./benchmark/data", k=5):
    # Read yaml file
    with open('./benchmark/parameters.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    options = []
    for name, params in config_data.items():
        options.append((name, params))
    # Evaluate all hyperparameter options
    output_path += "/hyperparameter_tuned_results"
    data_stream, n_instance_limit = load_data_stream(data_name, seed=seed, data_dir=data_dir)
    schema = data_stream.get_schema()
    for name, option in options:
        pool_models = []
        # combinations
        option = list(itertools.product(*(option.values())))
        for opt in option:
            if name.__eq__('sohot_options'):
                m = SoftHoeffdingTreeLayer(schema=schema, trees_num=1, max_depth=opt[0], ssp=opt[1], alpha=opt[2],
                                           seed=seed)
                output_path_c = f"{output_path}/SoHoT"
            elif name.__eq__('hoeffding_tree_options'):
                m = HoeffdingTree(schema=schema, grace_period=opt[0], confidence=opt[1], leaf_prediction=opt[2],
                                  random_seed=seed)
                output_path_c = f"{output_path}/HT"
            elif name.__eq__('hoeffding_tree_limit_options'):
                m = HoeffdingTreeLimit(schema=schema, grace_period=opt[0], confidence=opt[1], leaf_prediction=opt[2],
                                       random_seed=seed, node_limit=opt[3])
                output_path_c = f"{output_path}/HT_limit"
            elif name.__eq__('EFDT_options'):
                m = EFDT(schema=schema, grace_period=opt[0],
                         min_samples_reevaluate=opt[1], leaf_prediction=opt[2],
                         random_seed=seed)
                output_path_c = f"{output_path}/EFDT"
            elif name.__eq__('hat_options'):
                m = HoeffdingAdaptiveTree(schema=schema, grace_period=opt[0], confidence=opt[1],
                                          leaf_prediction=opt[2], random_seed=seed)
                output_path_c = f"{output_path}/HAT"
            elif name.__eq__('sgd_clf_options'):
                m = SGDClassifierMod(schema=schema, loss=opt[0], penalty=opt[1], learning_rate=opt[2]['lr'],
                                     eta0=opt[2]['eta0'], random_seed=seed)
                output_path_c = f"{output_path}/SGDClassifier"
            pool_models.append(m)
        model = ModelPool(models=pool_models, k=k)
        run_experiments(data_stream=data_stream, data_name=data_name, model=model, n_instance_limit=n_instance_limit,
                        seed=seed, output_path=output_path_c)


def set_ensemble(data_name, ensemble_size=10, seed=1, data_dir=".", output_path="./benchmark/data"):
    Path(f"{output_path}/ensemble_{ensemble_size}_results").mkdir(parents=True, exist_ok=True)
    data_stream, n_instance_limit = load_data_stream(data_name, seed=seed, data_dir=data_dir)
    schema = data_stream.get_schema()
    output_path += f"/ensemble_{ensemble_size}_results"
    models = [
        (SoftHoeffdingTreeLayer(schema=schema, trees_num=ensemble_size, seed=seed), f"{output_path}/SoHoT"),
        (AdaptiveRandomForestClassifier(schema=schema, ensemble_size=ensemble_size, random_seed=seed),
         f"{output_path}/ARF"),
        (StreamingRandomPatches(schema=schema, ensemble_size=ensemble_size, random_seed=seed), f"{output_path}/SRP"),
    ]
    for model, output_path in models:
        compute_complexity = output_path.endswith('SoHoT')
        run_experiments(data_stream=data_stream, data_name=data_name, model=model, n_instance_limit=n_instance_limit,
                        seed=seed, output_path=output_path, compute_complexity=compute_complexity)


def run_experiments(data_stream, data_name, model, n_instance_limit, seed, output_path, compute_complexity=False):
    data_stream.restart()
    tree_evaluator = TreeEvaluator(schema=data_stream.get_schema())
    i = 0
    while data_stream.has_more_instances():
        instance = data_stream.next_instance()
        # ----------- Test -----------
        y_pred = model.predict_proba(instance)
        # ----------- Train -----------
        model.train(instance)
        # ----------- Evaluate -----------
        tree_evaluator.update(y_pred=y_pred, y_target=instance.y_index)
        i += 1
        if i == n_instance_limit:
            break
    if compute_complexity:
        tree_evaluator.update_complexity(model.c_complexity())
    # ----------- Print to file -----------
    Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    open(f"{output_path}/{data_name}_summary-seed-{seed}.csv", "w").close()
    metrics = tree_evaluator.get_evaluation()
    print(metrics)
    df = pd.DataFrame(metrics, index=[0])
    df.to_csv(f"{output_path}/{data_name}_summary-seed-{seed}.csv",
              mode="a", index=False, header=True)


def plot_transparency(data_name, seed=1, data_dir="./data", n_instance_limit=None, visualize_tree_at=[], save_img=False):
    data_stream, n_instances = load_data_stream(data_name, seed=seed, data_dir=data_dir)
    if n_instance_limit is not None: n_instances = n_instance_limit
    schema = data_stream.get_schema()
    attribute_list = re.findall(r'@attribute\s+(\S+)', str(schema.get_moa_header()))

    model = SoftHoeffdingTreeLayer(schema=schema, seed=seed, trees_num=1)
    i = 0
    while data_stream.has_more_instances():
        instance = data_stream.next_instance()
        model.predict_proba(instance)
        model.train(instance)
        i += 1
        if i == n_instances:
            break
        if i in visualize_tree_at:
            x_transformed = torch.tensor(model.transform_input(instance), dtype=torch.float32, requires_grad=False)
            visualize_soft_hoeffding_tree(model.sohots[0], X=x_transformed, attribute_list=attribute_list,
                                          schema=schema, save_img=save_img, print_idx=i)

    # visualize_soft_hoeffding_tree(model.sohots[0], attribute_list=attribute_list)
