CUDA_VISIBLE_DEVICES=$1 \
python -u main.py --dataset last-fm --epoch 200 --dim 64 --lr 0.0001 --sim_regularity 0.0001 --batch_size 1024 --node_dropout True --node_dropout_rate 0.5 --mess_dropout True --mess_dropout_rate 0.1 --gpu_id 0 --context_hops 3 --K 3

