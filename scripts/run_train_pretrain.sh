
set -e


GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1  ./configs/pretrain.sh 

