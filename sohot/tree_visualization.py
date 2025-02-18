# Graph Visualization of Soft Hoeffding Trees
# Bibliography: NetworkX
import networkx as nx
from .internal_node import Node
from .leaf_node import LeafNode
import matplotlib.pyplot as plt
import random
from pathlib import Path

'''
Note: Edge probabilities are visualized with 3 decimals, therefore probability 1 is not always 1.
    Color: #8FAADC
'''


def get_fi_impact(sohot, node, x, attribute_list, schema):
    if x is None:
        return ""
    w_i = sohot.weights.get(node.orientation_sequence)
    x_dot_w_outer = [abs(valu) for valu in (w_i * x)]
    x_dot_w_inner = abs(sum(x_dot_w_outer))
    percentage_feature_impact = [sohot.alpha * (x_dot_w_outer[j] / x_dot_w_inner).detach().item() for j in
                                 range(len(x_dot_w_outer))]
    num_features = len(w_i)
    average_percentage = 1 / num_features
    impact_str = ""

    if schema is not None:
        feature_one_hot_offset = 0
        for i in range(schema.get_num_attributes()):
            # Verify if feature is one hot encoded
            if schema.get_moa_header().attribute(i).isNominal() \
                    and schema.get_moa_header().attribute(i).numValues() > 2:
                len_attribute_values = schema.get_moa_header().attribute(i).numValues()
                start, end = i + feature_one_hot_offset, i + feature_one_hot_offset + len_attribute_values
                # impact = max(percentage_feature_impact[start:end])                    # max
                impact = sum(percentage_feature_impact[start:end]) / (end-start)        # average
                feature_one_hot_offset += len_attribute_values - 1
            else:
                impact = percentage_feature_impact[i + feature_one_hot_offset]
            # Add feature as important if impact is high
            if impact >= average_percentage:
                impact_str += f"\n{attribute_list[i]}:{impact:3f}"
    return impact_str


def visualize_soft_hoeffding_tree(sohot, X=None, print_idx=0, save_img=False, attribute_list={}, schema=None):
    if save_img:
        Path("data/images_sohot").mkdir(parents=True, exist_ok=True)
    G = nx.Graph()
    node_label = {}
    edge_label = {}
    G.add_node(sohot.root)
    no_edge_labels = True       # do not print edge labels
    font_size = 8
    if X is None:
        no_edge_labels = True

    # --------------- Tree has only one node ---------------
    if isinstance(sohot.root, LeafNode):
        # Print weight? w: {sohot.weights[sohot.root.orientation_sequence].data.numpy()}, \n
        node_label[sohot.root] = f"P(x->l)={sohot.root.sample_to_node_prob:.2f}"
        pos = nx.spring_layout(G)
        label_shift = 0.01

    # --------------- Traverse whole tree ---------------
    else:
        if len(attribute_list) == 0:
            node_label[sohot.root] = "Split Attr: {}, \nValue:{:.3f}".format(sohot.root.split_test.feature,
                                                                             sohot.root.split_test.split_at)
        else:
            impact = get_fi_impact(sohot, sohot.root, X, attribute_list=attribute_list, schema=schema)
            node_label[sohot.root] = f"{attribute_list[sohot.root.split_test.feature]} > {sohot.root.split_test.split_at:.3f}{impact}"
        if sohot.root.right_leaf is None:
            to_traverse = [sohot.root.right]
        else:
            to_traverse = [sohot.root.right_leaf]
        if sohot.root.left_leaf is None:
            to_traverse.append(sohot.root.left)
        else:
            to_traverse.append(sohot.root.left_leaf)
        previous = [sohot.root, sohot.root]

        while to_traverse:
            i = to_traverse.pop()
            prev = previous.pop()
            G.add_node(i)
            G.add_edge(prev, i)
            orientation_seq = prev.orientation_sequence
            weight_vec = sohot.weights[orientation_seq]

            if isinstance(i, Node):
                impact = get_fi_impact(sohot, i, X, attribute_list=attribute_list, schema=schema)

                if len(attribute_list) == 0:
                    node_label[i] = "Split Attr: {}, \nValue:{:.3f}".format(i.split_test.feature, i.split_test.split_at)
                else:
                    node_label[i] = f"{attribute_list[i.split_test.feature]} > {i.split_test.split_at:.3f}{impact}"
                if not no_edge_labels:
                    if prev.left is i:
                        edge_label[(prev, i)] = "{:.6f}".format(prev.forward(X, weight_vec))
                    else:
                        edge_label[(prev, i)] = "{:.6f}".format(1. - prev.forward(X, weight_vec))
                if i.right_leaf is None:
                    to_traverse.append(i.right)
                else:
                    to_traverse.append(i.right_leaf)
                if i.left_leaf is None:
                    to_traverse.append(i.left)
                else:
                    to_traverse.append(i.left_leaf)
                previous.append(i)
                previous.append(i)
            else:
                node_label[i] = "P(x->l)={:.2f}".format(i.sample_to_node_prob)

        pos = hierarchy_pos(G, sohot.root)
        label_shift = 0.05

    ax = plt.gca()
    ax.set_title(f"Soft Hoeffding Tree, # Instances: {print_idx}")
    nx.draw(G, pos, node_color='#8FAADC', with_labels=False)
    label_pos = {node: (x + label_shift, y) for node, (x, y) in pos.items()}  # Move right by label_shift
    nx.draw_networkx_labels(G, label_pos, labels=node_label, font_size=font_size,
                            verticalalignment='center', horizontalalignment='left',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    if not no_edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label, font_size=font_size)
    # Enlarge the figure to show all labels
    x_values, y_values = zip(*pos.values())
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    plt.xlim(x_min - 0.05, x_max + label_shift + 0.15)
    plt.ylim(y_min - 0.05, y_max + 0.1)
    plt.margins(0.05)
    # plt.tight_layout()
    if save_img:
        plt.savefig(f"data/images_sohot/{print_idx}")
        plt.clf()
    else:
        plt.show()


# source: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
