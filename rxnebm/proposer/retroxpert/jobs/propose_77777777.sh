#!/bin/bash
#SBATCH -n 1                                # number of comands issued in batch script at any times
#SBATCH --cpus-per-task=8                  # number of cores to allocate each task
#SBATCH -N 1                                # Request 1 node
#SBATCH --time=24:00:00

#SBATCH --gres=gpu:1                      # specify GPU count
#SBATCH -p sched_mit_ccoley                 # sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=8000                  # Request 4G of memory per CPU

#SBATCH -o logs/propose_output_%x_%j.txt               # redirect output to output_JOBNAME_JOBID.txt
#SBATCH -e logs/propose_error_%x_%j.txt                # redirect errors to error_JOBNAME_JOBID.txt
#SBATCH -J propose_77777777                           # name of job
#SBATCH --mail-type=ALL               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

seed=77777777

printf "Running EGAT inference on valid\n"
python3 EGAT_valid.py --load --seed ${seed}

printf "Running prepare_data_train_noaug.py\n"
python3 prepare_data_train_noaug.py --seed ${seed}

printf "Running prepare_test_prediction_all.py\n"
python3 prepare_test_prediction_all.py --seed ${seed}

printf "Running bash translate_all.sh\n"
cd OpenNMT-py/script/USPTO50K-aug-untyped
bash translate_all.sh ${seed}

printf "Running score_all.sh\n"
bash score_all.sh ${seed}