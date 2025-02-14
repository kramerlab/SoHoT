import torch.nn as nn
import torch
from .sohot import SoftHoeffdingTree
import math
from river import stats
import random


class SoftHoeffdingTreeLayer(nn.Module):
    def __init__(self, schema, ssp=1.0, max_depth=7, trees_num=10, split_confidence=1e-6,
                 tie_threshold=0.05, grace_period=600, is_target_class=True, average_output=False,
                 seed=None, alpha=0.3, optimizer='adam', lr=0.01):
        super(SoftHoeffdingTreeLayer, self).__init__()
        self.schema = schema
        # Compute one-hot-encoded input dimension using schema
        self.num_attributes = self.schema.get_num_attributes()
        self.input_dim = 0
        for i in range(self.num_attributes):
            if self.schema.get_moa_header().attribute(i).isNominal() \
                    and self.schema.get_moa_header().attribute(i).numValues() > 2:
                len_attribute_values = self.schema.get_moa_header().attribute(i).numValues()
                self.input_dim += len_attribute_values
            else:
                self.input_dim += 1
        self.output_dim = self.schema.get_num_classes()

        # Normalization (instead of BachNorm)
        self.use_normalization = True
        self.means = [stats.Mean() for _ in range(self.num_attributes)]
        self.variances = [stats.Var() for _ in range(self.num_attributes)]

        if seed is None:
            seeds = [None] * trees_num
        else:
            torch.manual_seed(seed)
            random.seed(seed)
            seeds = [random.random() for _ in range(trees_num)]

        # self.bn = nn.BatchNorm1d(self.input_dim, momentum=0.99, eps=0.001)       # BN adds randomness
        self.sohots = nn.ModuleList([SoftHoeffdingTree(self.input_dim, self.output_dim, max_depth=max_depth,
                                                       smooth_step_param=ssp, alpha=alpha,
                                                       split_confidence=split_confidence,
                                                       tie_threshold=tie_threshold, grace_period=grace_period,
                                                       seed=seeds[t])
                                     for t in range(trees_num)])

        # Initialize an Optimizer
        self.lr = lr
        if optimizer.__eq__('adam'):
            self.optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.optim.zero_grad()
        self.old_params = sum(1 for _ in self.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=-1)

        self.is_target_class = is_target_class
        self.average_output = average_output

    def forward(self, x, y=None):
        # x = self.bn(x)
        outputs = [None]*len(self.sohots)
        for i, shtree in enumerate(self.sohots):
            outputs[i] = shtree(x, y)
        if self.average_output:
            x = torch.stack(outputs, dim=0).mean(dim=0)
        else:
            x = torch.stack(outputs, dim=0).sum(dim=0)
        return self.softmax(x)

    def _get_normalized_value(self, value, index):
        self.variances[index].update(value)
        variance = self.variances[index].get()
        self.means[index].update(value)
        mean = self.means[index].get()
        sd = math.sqrt(variance)
        if sd > 0.:
            return (value - mean) / (3. * sd)
        else:
            return 0.

    def transform_input(self, instance):
        # Pipeline: CapyMOA instance -> one-hot (categorical) and feature normalization (numerical) -> PyTorch tensor
        x = instance.x
        x_trans = []
        for i in range(self.num_attributes):
            value = x[i]
            if self.schema.get_moa_header().attribute(i).isNominal() \
                    and self.schema.get_moa_header().attribute(i).numValues() > 2:
                # Do one hot encoding
                len_attribute_values = self.schema.get_moa_header().attribute(i).numValues()
                one_hot_encoded_attribute = [0.] * len_attribute_values
                one_hot_encoded_attribute[int(value)] = 1.
                x_trans.extend(one_hot_encoded_attribute)
            elif self.use_normalization:
                x_trans.append(self._get_normalized_value(value, i))
            else:
                x_trans.append(value)
        return x_trans

    def predict_proba(self, instance):
        x_trans = self.transform_input(instance)
        return self.forward(torch.tensor([x_trans], dtype=torch.float32),
                            torch.tensor([instance.y_index], dtype=torch.long))

    def predict(self, instance):
        return torch.argmax(self.predict_proba(instance)).item()

    def train(self, instance):
        # todo perform forward pass and store information - does it make sense for larger batch sizes?
        y_pred = self.predict_proba(instance)
        y_target = torch.tensor([instance.y_index], dtype=torch.long)
        # If the number of parameters has changed, update the optimizer such that the new weight
        # parameter are registered and also updated in the backward pass
        if self.old_params != sum(1 for _ in self.parameters()):
            self.optim = torch.optim.Adam(self.parameters(), lr=self.lr)
            self.old_params = sum(1 for _ in self.parameters())
        loss = self.criterion(y_pred, y_target)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

    def c_complexity(self):
        return [t.total_node_cnt() for t in self.sohots]
