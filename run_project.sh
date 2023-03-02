#!/bin/bash

pip install simplet5 -q

pip install sklearn -q

pip install datasets -q

pip install pandas -q

# set the number of nodes
SBATCH --nodes=1

# set max wallclock time
SBATCH --time=10:00:00

# set name of job
SBATCH --job-name=t5-cot-base-test

# set number of GPUs
SBATCH --gres=gpu:2

# mail alert at start, end and abortion of execution
SBATCH --mail-type=ALL

# send mail to this address
SBATCH --mail-user=dg707@bath.ac.uk

python3 main.py