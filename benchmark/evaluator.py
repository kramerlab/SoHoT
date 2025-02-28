from capymoa.evaluation import ClassificationEvaluator, ClassificationWindowedEvaluator
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from moa.core import Utils
import warnings
warnings.filterwarnings("ignore")


class TreeEvaluator:
    def __init__(self, schema):
        self.schema = schema
        self.label_indices = schema.get_label_indexes()
        self.num_instances = 0
        self.clf_evaluator = ClassificationEvaluator(schema=schema)
        self.clf_windowed_evaluator = ClassificationWindowedEvaluator(schema=schema, window_size=1000)
        # AUROC
        self.y_pred_list = []
        self.y_target_list = []
        self.positive_class = 1
        # Cross entropy loss
        self.sum_cross_entropy_loss = 0
        # Complexities
        self.tree_complexities = []
        self.sum_tree_complexities = []

    def update(self, y_pred, y_target, tree_complexity=None):
        self.num_instances += 1
        # Differentiate between SoHoT output and CapyMOA output
        if isinstance(y_pred, torch.Tensor):
            y_pred_index = torch.argmax(y_pred[0]).item()
            y_pred = y_pred[0].detach().numpy()
        else:
            y_pred = list(y_pred)
            # First prediction special cases: y_pred for SGD is None, for HT weird behavior happens
            if y_pred is None \
                    or len(y_pred) != len(self.label_indices) \
                    or any(np.isnan(y_pred_v) for y_pred_v in y_pred) \
                    or sum(y_pred) == 0:
                # if not all classes has been discovered, add zero for unobserved classes
                if y_pred is not None \
                        and len(y_pred) >= 2 \
                        and sum(y_pred) != 0 \
                        and not any(np.isnan(y_pred_v) for y_pred_v in y_pred):
                    while len(y_pred) != len(self.label_indices):
                        y_pred.append(0.)
                else:
                    y_pred = [0.] * max(1, len(self.label_indices))
                    y_pred[0] = 1.
            # predict_proba returns class votes not probabilities
            if sum(y_pred) != 1:
                y_pred = np.array([votes / sum(y_pred) for votes in y_pred])
            y_pred_index = Utils.maxIndex(y_pred)
        self.clf_evaluator.update(y_target_index=y_target, y_pred_index=y_pred_index)
        self.clf_windowed_evaluator.update(y_target_index=y_target, y_pred_index=y_pred_index)
        # Area under roc curve
        self.y_pred_list.append(y_pred)
        self.y_target_list.append(y_target)
        # Cross entropy loss
        # print(f"y_pred: {np.array([y_pred])}, y_target: {np.array([y_target])}")
        self.sum_cross_entropy_loss += log_loss(y_pred=np.array([y_pred]), y_true=np.array([y_target]),
                                                labels=self.label_indices)
        if tree_complexity is not None:
            if not self.sum_tree_complexities:
                self.sum_tree_complexities = tree_complexity
            else:
                self.sum_tree_complexities = [sum(tc) for tc in zip(self.sum_tree_complexities, tree_complexity)]

    def update_complexity(self, tree_complexities):
        self.tree_complexities = tree_complexities

    def get_auroc(self):
        if self.schema.get_num_classes() > 2:
            auroc = roc_auc_score(self.y_target_list, self.y_pred_list, multi_class='ovr')
        else:
            auroc = roc_auc_score(self.y_target_list, np.array(self.y_pred_list)[:, self.positive_class])
        return auroc

    def get_evaluation(self):
        metrics = {'Accuracy': self.clf_evaluator.accuracy(),
                   'F1': self.clf_evaluator.f1_score(),
                   'Auroc': self.get_auroc(),
                   'Cross Entropy Loss': self.sum_cross_entropy_loss / self.num_instances,
                   }
        for i, complexity in enumerate(self.tree_complexities):
            metrics[f'Tree {i} Complexity'] = complexity
        for i, complexity in enumerate(self.sum_tree_complexities):
            metrics[f'Avg Tree {i} Complexity'] = complexity / self.num_instances
        return metrics

    def get_window_evaluation(self):
        return {'Accuracy Window': self.clf_windowed_evaluator.accuracy(),
                'F1 Window': self.clf_windowed_evaluator.f1_score(),
                }

# import numpy as np
#
#
# # Gama et al.: 1000, 2000, 3000, 4000, 5000 (PAW: w=1000)
# def prequential_error(losses, window_size=1000):
#     l_slided = []
#     l_window = losses[:window_size]
#     for i_l in losses[window_size:]:
#         l_slided.append(np.mean(l_window))
#         l_window.pop(0)
#         l_window.append(i_l)
#     return l_slided
#
#
# # Prequential error computed at time i (=len(losses)), with fading factor alpha
# def fading_prequential_error(losses, fading_factor=0.995):
#     time_elapsed = len(losses)
#     dividend, divisor = 0, 0
#     for k in range(1, time_elapsed+1):
#         dividend += (fading_factor**(time_elapsed-k)) * losses[k-1]
#         divisor += fading_factor ** (time_elapsed - k)
#     return dividend/divisor
#
#
# def fading_prequential_over_time(losses, fading_factor=0.995):
#     fading_losses = []
#     for i in range(len(losses)):
#         fading_losses.append(fading_prequential_error(losses[:i+1], fading_factor=fading_factor))
#     return fading_losses
