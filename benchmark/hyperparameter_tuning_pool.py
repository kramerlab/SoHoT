from capymoa.drift.detectors import ADWIN
import numpy as np
from moa.core import Utils


class ModelPool:
    def __init__(self, models, k):
        self.pool = [(model, ADWIN(delta=1e-5)) for model in models]
        self.pool_size = len(models)
        self.k = k

    @staticmethod
    def _get_performance(pool_entry):
        return pool_entry[1].moa_detector.getEstimation()

    def _update_detectors(self, instance):
        for model, detector in self.pool:
            y_pred = model.predict(instance)
            detector.add_element(instance.y_index == y_pred)

    def _get_performance_sorted_indices(self):
        return np.argsort([self._get_performance(p) for p in self.pool])

    def predict(self, instance):
        return Utils.maxIndex(self.predict_proba(instance))

    def predict_proba(self, instance):
        best_model_idx = self._get_performance_sorted_indices()[-1]
        return self.pool[best_model_idx][0].predict_proba(instance)

    def train(self, instance):
        # Update the performance detectors
        self._update_detectors(instance)
        # 1. Select the top models and 2. Randomly choose the remaining models to train
        models_to_be_trained = [0] * self.pool_size
        num_top_models_to_train = self.k // 2
        top_models_to_train = self._get_performance_sorted_indices()[:num_top_models_to_train]
        rand_models_to_train = np.random.choice([i for i in range(self.pool_size) if i not in top_models_to_train],
                                                size=self.k - num_top_models_to_train, replace=False)
        for pool_idx in range(self.pool_size):
            if pool_idx in top_models_to_train or pool_idx in rand_models_to_train:
                models_to_be_trained[pool_idx] = 1
        # Train the models
        for i, (model, _) in enumerate(self.pool):
            if models_to_be_trained[i] == 1:
                model.train(instance)
