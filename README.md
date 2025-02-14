# Soft Hoeffding Tree

This repository contains a new predictive base learner Soft Hoeffding Tree (SoHoT) for large and drifting data streams.
A SoHoT combines the extensibility and transparency of Hoeffding trees with the differentiability of soft trees.
A new gating function is used to regulate the balance between transparent and soft routing in a SoHoT, 
which is essentially a trade-off between transparency and performance. 
SoHoT can be used in a neural network due to its differentiability.



## Example Code
```python

# Visualize the first tree in the ensemble
# visualize_soft_hoeffding_tree(sohotel.sohots[0])
```