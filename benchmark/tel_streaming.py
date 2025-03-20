# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Tree Ensemble Layer ---------------------------------------------

# Note: TEL need to be installed manually (Change sys path to TreeEnsembleLayer's current location)
#       Works only with Tensorflow version 2.9.0
#   See: https://github.com/google-research/google-research/tree/master/tf_trees
import sys
sys.path.insert(1, '/home/user/TreeEnsembleLayer')

from tf_trees import TEL
import tensorflow as tf
import numpy as np
from river import stats
import math


class TreeEnsembleLayerStreaming:
    def __init__(self, schema, trees_num=10, depth=7, smooth_step_param=1., seed=42, learning_rate=0.001,
                 use_normalization=True, measure_transparency=False):
        self.schema = schema
        self.num_attributes = self.schema.get_num_attributes()
        self.output_logits_dim = self.schema.get_num_classes()

        self.trees_num = trees_num
        self.depth = depth
        self.smooth_step_param = smooth_step_param
        self.learning_rate = learning_rate
        # Evaluate batch size 1
        self.batch_size = 1

        tf.random.set_seed(seed)
        tree_layer = TEL(output_logits_dim=self.output_logits_dim, trees_num=trees_num, depth=depth,
                         smooth_step_param=smooth_step_param)
        self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tree_layer)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss_func = tf.nn.softmax_cross_entropy_with_logits
        self.model.compile(loss=loss_func, optimizer=optimizer)

        # Feature standardization (instead of BatchNorm)
        self.use_normalization = use_normalization
        self.means = [stats.Mean() for _ in range(self.num_attributes)]
        self.variances = [stats.Var() for _ in range(self.num_attributes)]

        self.measure_transparency = measure_transparency
        self.total_n_important_feat = 0
        self.n_instances_seen = 0

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
        x = instance.x
        x_trans = []
        for i in range(self.num_attributes):
            value = x[i]
            if self.schema.get_moa_header().attribute(i).isNominal() \
                    and self.schema.get_moa_header().attribute(i).numValues() > 2:
                # One hot encoding
                len_attribute_values = self.schema.get_moa_header().attribute(i).numValues()
                one_hot_encoded_attribute = [0.] * len_attribute_values
                one_hot_encoded_attribute[int(value)] = 1.
                x_trans.extend(one_hot_encoded_attribute)
            elif self.use_normalization:
                x_trans.append(self._get_normalized_value(value, i))
            else:
                x_trans.append(value)
        return np.array([x_trans])

    def transform_label(self, instance):
        y_transform = [0.] * self.output_logits_dim
        y_transform[instance.y_index] = 1.
        return np.array([y_transform])

    # Measures the length of the explanation for each sample routing (need to be called after training)
    def _measure_length_explanation(self, x):
        self.n_instances_seen += 1
        n_relevant_features = 0
        weights = self.model.layers[0]._node_weights[0].numpy()
        num_weights = tf.shape(weights)[1]
        for x_val in x:
            for c_vec in range(num_weights):
                w_i = weights[:, c_vec]
                x_dot_w_outer = [abs(val) for val in (x_val * w_i)]
                x_dot_w_inner = abs(np.sum(x_dot_w_outer))
                percentage_feature_impact = [x_dot_w_outer[j] / x_dot_w_inner for j in range(len(x_dot_w_outer))]
                num_features = len(w_i)
                average_percentage = 1 / num_features
                impact = sum([1 if impact >= average_percentage else 0 for impact in percentage_feature_impact])
                n_relevant_features += impact
        self.total_n_important_feat += n_relevant_features

    def get_avg_explanation_len(self):
        return self.total_n_important_feat / (self.n_instances_seen * (pow(2, self.depth) - 1))

    def predict(self, instance):
        return np.argmax(self.predict_proba(instance))

    def predict_proba(self, instance):
        x = self.transform_input(instance)
        y_pred = self.model(x, training=False)
        if self.batch_size == 1:
            return tf.nn.softmax(y_pred)[0].numpy()
        else:
            return tf.nn.softmax(y_pred).numpy()

    def train(self, instance):
        x = self.transform_input(instance)
        y = self.transform_label(instance)
        self.model.train_on_batch(x, y)
        # Track length of explanation
        if self.measure_transparency:
            self._measure_length_explanation(x)
