**KGIL: Unleashing the Power of Knowledge Graph for Recommendation via Invariant Learning**

## Environment Requirement

The code has been tested running under Python 3.6.5. The required packages are as follows:

- pytorch == 1.5.0
- numpy == 1.15.4
- scipy == 1.1.0
- sklearn == 0.20.0
- torch_scatter == 2.0.5
- networkx == 2.5

## Reproducibility & Example to Run the Codes


- Last-fm dataset

```
python -u main.py --dataset last-fm --epoch 200 --dim 64 --lr 0.0001 --sim_regularity 0.0001 --batch_size 1024 --node_dropout True --node_dropout_rate 0.5 --mess_dropout True --mess_dropout_rate 0.1 --gpu_id 0 --context_hops 3 --K 3

```


