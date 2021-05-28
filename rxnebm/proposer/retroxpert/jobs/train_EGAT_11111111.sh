#!/bin/bash
#SBATCH -n 1                                # number of comands issued in batch script at any times
#SBATCH --cpus-per-task=8                  # number of cores to allocate each task
#SBATCH -N 1                                # Request 1 node
#SBATCH --time=10:00:00

#SBATCH --gres=gpu:1                       # specify GPU count
#SBATCH -p sched_mit_ccoley                 # sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=8000                  # Request 4G of memory per CPU

#SBATCH -o logs/EGAT_output_%x_%j.txt               # redirect output to output_JOBNAME_JOBID.txt
#SBATCH -e logs/EGAT_error_%x_%j.txt                # redirect errors to error_JOBNAME_JOBID.txt
#SBATCH -J EGAT_11111111                            # name of job
#SBATCH --mail-type=ALL               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

seed=11111111

mkdir OpenNMT-py/data/USPTO50K-aug-untyped_${seed}
cp copy_files.py OpenNMT-py/data/USPTO50K-aug-untyped_${seed}/

mkdir OpenNMT-py/experiments/
mkdir OpenNMT-py/experiments/results_${seed}
mkdir OpenNMT-py/experiments/checkpoints/USPTO50K-aug-untyped_${seed}
cp average_models.sh OpenNMT-py/experiments/checkpoints/USPTO50K-aug-untyped_${seed}/

printf "running train.py\n"
python3 train.py --seed ${seed}

printf "running train.py with --test_only --load\n"
python3 train.py --test_only --load --seed ${seed}

printf "running train.py with --test_on_train --load\n"
python3 train.py --test_on_train --load --seed ${seed}

# need seed bcos we are generating non-cano SMILES
printf "running prepare_data.py\n"
python3 prepare_data.py --seed ${seed}

printf "running prepare_test_prediction.py\n"
python3 prepare_test_prediction.py --seed ${seed}

printf "running prepare_train_error_aug.py\n"
python3 prepare_train_error_aug.py --seed ${seed}
