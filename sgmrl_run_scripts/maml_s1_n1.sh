#!/bin/bash

#SBATCH -o ../logs/log-%j
#SBATCH -c 10

source /etc/profile

python ../sgmrl_experiments/vpg_run.py --algo maml \
                                    --seed 1 \
                                    --n_adapt_steps 1
