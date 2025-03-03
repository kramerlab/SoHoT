from benchmark.evaluator import TreeEvaluator
from benchmark.load_data import load_data_stream
from benchmark.hyperparameter_tuning_pool import ModelPool
from sohot.sohot_ensemble_layer import SoftHoeffdingTreeLayer
from capymoa.classifier import HoeffdingTree, EFDT, HoeffdingAdaptiveTree, AdaptiveRandomForestClassifier, \
    StreamingRandomPatches
from benchmark.helpers import SGDClassifierMod, HoeffdingTreeLimit, HoeffdingAdaptiveTreeMod, HoeffdingTreeRiver, \
    HoeffdingAdaptiveTreeRiver
from pathlib import Path
import pandas as pd
import yaml
import itertools


# Note: Epsilon data is not working for CapyMOA's Hoeffding Tree variants (due to the amount of attributes and
# normalization). Use River's Hoeffding tree instead!


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
        compute_complexity = False
        # combinations
        option = list(itertools.product(*(option.values())))
        for opt in option:
            if name.__eq__('sohot_options'):
                m = SoftHoeffdingTreeLayer(schema=schema, trees_num=1, max_depth=opt[0], ssp=opt[1], alpha=opt[2],
                                           seed=seed)
                output_path_c = f"{output_path}/SoHoT"
                compute_complexity = True
            elif name.__eq__('hoeffding_tree_options'):
                if data_name.__eq__('epsilon'):
                    m = HoeffdingTreeRiver(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                           leaf_prediction=opt[2], random_seed=seed)
                else:
                    m = HoeffdingTree(schema=schema, grace_period=opt[0], confidence=float(opt[1]), leaf_prediction=opt[2],
                                      random_seed=seed)
                output_path_c = f"{output_path}/HT"
            elif name.__eq__('hoeffding_tree_limit_options'):
                if data_name.__eq__('epsilon'):
                    m = HoeffdingTreeRiver(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                           leaf_prediction=opt[2], random_seed=seed, limit=opt[4])
                else:
                    m = HoeffdingTreeLimit(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                           leaf_prediction=opt[2], random_seed=seed, node_limit=opt[3])
                output_path_c = f"{output_path}/HT_limit"
            elif name.__eq__('EFDT_options'):
                m = EFDT(schema=schema, grace_period=opt[0],
                         min_samples_reevaluate=opt[1], leaf_prediction=opt[2],
                         random_seed=seed)
                output_path_c = f"{output_path}/EFDT"
            elif name.__eq__('hat_options'):
                if data_name.__eq__('epsilon'):
                    m = HoeffdingAdaptiveTreeRiver(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                                   leaf_prediction=opt[2], random_seed=seed)
                else:
                    m = HoeffdingAdaptiveTreeMod(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                                 leaf_prediction=opt[2], random_seed=seed)
                output_path_c = f"{output_path}/HAT"
                compute_complexity = True
            elif name.__eq__('sgd_clf_options'):
                m = SGDClassifierMod(schema=schema, loss=opt[0], penalty=opt[1], learning_rate=opt[2]['lr'],
                                     eta0=opt[2]['eta0'], random_seed=seed)
                output_path_c = f"{output_path}/SGDClassifier"
            pool_models.append(m)
        model = ModelPool(models=pool_models, k=k)
        run_experiments(data_stream=data_stream, data_name=data_name, model=model, n_instance_limit=n_instance_limit,
                        seed=seed, output_path=output_path_c, compute_complexity=compute_complexity)


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
        tree_evaluator.update(y_pred=y_pred, y_target=instance.y_index,
                              tree_complexity=model.c_complexity() if compute_complexity else None)
        i += 1
        if n_instance_limit is not None and i == n_instance_limit:
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


def plot_transparency(data_name, seed=1, data_dir="./data", n_instance_limit=None, visualize_tree_at=[],
                      save_img=False):
    data_stream, n_instances = load_data_stream(data_name, seed=seed, data_dir=data_dir)
    if n_instance_limit is not None: n_instances = n_instance_limit
    schema = data_stream.get_schema()

    model = SoftHoeffdingTreeLayer(schema=schema, seed=seed, trees_num=1, lr=0.08)
    i = 0
    while data_stream.has_more_instances():
        instance = data_stream.next_instance()
        model.sohots[0]._reset_all_node_to_leaf_prob()
        y_pred = model.predict_proba(instance)

        if i in visualize_tree_at:
            print(f"Prediction at {i}: {y_pred.detach().numpy()}, true label: {instance.y_index}")
            model.plot_tree(instance=instance, tree_idx=0, save_img=save_img)

        model.train(instance)
        if i == n_instances:
            break
        i += 1
