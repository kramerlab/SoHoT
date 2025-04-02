from capymoa.classifier import HoeffdingTree, SGDClassifier, HoeffdingAdaptiveTree, EFDT
from capymoa.instance import Instance
from capymoa.splitcriteria import SplitCriterion
from capymoa.stream import Schema
from typing import Union
import river
import re


# Note: - Epsilon data is not working for CapyMOA's Hoeffding Tree variants (due to the amount of attributes and
#         normalization). Use River's Hoeffding tree instead!
#       - In CapyMOA Version 0.8 SGDClassifier does not return anything!
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
        tree_size = abs(self.moa_learner.getNodeCount())
        if tree_size > self.node_limit:
            self.moa_learner.gracePeriodOption.setValue(10000000)
            # self.moa_learner.growth_limit = False     # protected attribute cannot be reached
        super().train(instance)

    def c_complexity(self):
        return self.moa_learner.getNodeCount()


class HoeffdingTreeMod(HoeffdingTree):
    def c_complexity(self):
        return self.moa_learner.getNodeCount()


class HoeffdingAdaptiveTreeMod(HoeffdingAdaptiveTree):
    def c_complexity(self):
        return self.moa_learner.getNodeCount()


# Epsilon data has too many feature such that internal normalization fails, therefore use Hoeffding Tree from River
class HoeffdingTreeRiver(river.tree.HoeffdingTreeClassifier):
    def __init__(self, schema, grace_period, confidence, leaf_prediction, random_seed, limit=None):
        self.input_dim = int(schema.get_num_attributes())
        nominal_attributes = [i for i in range(self.input_dim)
                              if schema.get_moa_header().attribute(i).isNominal()]
        leaf_preds = {'MajorityClass': 'mc', 'NaiveBayes': 'nb', 'NaiveBayesAdaptive': 'nba'}
        self.ht = river.tree.HoeffdingTreeClassifier(grace_period=grace_period, delta=confidence,
                                                     nominal_attributes=nominal_attributes,
                                                     leaf_prediction=leaf_preds[leaf_prediction],
                                                     max_depth=limit)

    def river_input_format(self, x):
        return dict(zip([i for i in range(self.input_dim)], x))

    def predict(self, instance: Instance):
        x = self.river_input_format(instance.x)
        return self.ht.predict_one(x=x)

    def predict_proba(self, instance: Instance):
        x = self.river_input_format(instance.x)
        # predict_proba_one returns a dictionary that associates a probability which each label.
        return list(self.ht.predict_proba_one(x).values())

    def train(self, instance: Instance):
        x = self.river_input_format(instance.x)
        self.ht.learn_one(x=x, y=instance.y_index)

    def c_complexity(self):
        return self.ht.n_nodes


class HoeffdingAdaptiveTreeRiver(river.tree.HoeffdingAdaptiveTreeClassifier):
    def __init__(self, schema, grace_period, confidence, leaf_prediction, random_seed):
        self.input_dim = int(schema.get_num_attributes())
        nominal_attributes = [i for i in range(self.input_dim)
                              if schema.get_moa_header().attribute(i).isNominal()]
        leaf_preds = {'MajorityClass': 'mc', 'NaiveBayes': 'nb', 'NaiveBayesAdaptive': 'nba'}
        self.hat = river.tree.HoeffdingAdaptiveTreeClassifier(grace_period=grace_period, delta=confidence,
                                                             nominal_attributes=nominal_attributes,
                                                             leaf_prediction=leaf_preds[leaf_prediction],
                                                             seed=random_seed)

    def river_input_format(self, x):
        return dict(zip([i for i in range(self.input_dim)], x))

    def predict(self, instance: Instance):
        x = self.river_input_format(instance.x)
        return self.hat.predict_one(x=x)

    def predict_proba(self, instance: Instance):
        x = self.river_input_format(instance.x)
        return list(self.hat.predict_proba_one(x).values())

    def train(self, instance: Instance):
        x = self.river_input_format(instance.x)
        self.hat.learn_one(x=x, y=instance.y_index)

    def c_complexity(self):
        return self.hat.n_nodes


class EFDTMod(EFDT):
    def c_complexity(self):
        n_nodes_str = str(self.moa_learner.getModelMeasurements()[2])
        n_nodes = int(re.search(r"\d+", n_nodes_str).group())
        return n_nodes


class EFDTRiver(river.tree.ExtremelyFastDecisionTreeClassifier):
    def __init__(self, schema, grace_period, min_samples_reevaluate, leaf_prediction, random_seed):
        self.input_dim = int(schema.get_num_attributes())
        nominal_attributes = [i for i in range(self.input_dim)
                              if schema.get_moa_header().attribute(i).isNominal()]
        leaf_preds = {'MajorityClass': 'mc', 'NaiveBayes': 'nb', 'NaiveBayesAdaptive': 'nba'}
        self.efdt = river.tree.ExtremelyFastDecisionTreeClassifier(grace_period=grace_period,
                                                                   min_samples_reevaluate=min_samples_reevaluate,
                                                                   nominal_attributes=nominal_attributes,
                                                                   leaf_prediction=leaf_preds[leaf_prediction])

    def river_input_format(self, x):
        return dict(zip([i for i in range(self.input_dim)], x))

    def predict(self, instance: Instance):
        x = self.river_input_format(instance.x)
        return self.efdt.predict_one(x=x)

    def predict_proba(self, instance: Instance):
        x = self.river_input_format(instance.x)
        return list(self.efdt.predict_proba_one(x).values())

    def train(self, instance: Instance):
        x = self.river_input_format(instance.x)
        self.efdt.learn_one(x=x, y=instance.y_index)

    def c_complexity(self):
        return self.efdt.n_nodes
