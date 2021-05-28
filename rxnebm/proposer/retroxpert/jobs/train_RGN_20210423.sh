#!/bin/bash
#SBATCH -n 1                                # number of comands issued in batch script at any times
#SBATCH --cpus-per-task=8                  # number of cores to allocate each task
#SBATCH -N 1                                # Request 1 node
#SBATCH --time=35:00:00

#SBATCH --gres=gpu:2                      # specify GPU count
#SBATCH -p sched_mit_ccoley                 # sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=8000                  # Request 4G of memory per CPU

#SBATCH -o logs/RGN_output_%x_%j.txt               # redirect output to output_JOBNAME_JOBID.txt
#SBATCH -e logs/RGN_error_%x_%j.txt                # redirect errors to error_JOBNAME_JOBID.txt
#SBATCH -J RGN_20210423                          # name of job
#SBATCH --mail-type=ALL               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

printf "Running bash preprocess.sh\n"
cd OpenNMT-py/script/USPTO50K-aug-untyped
bash preprocess.sh 20210423

printf "Running bash train.sh\n"
bash train.sh 20210423

printf "Running bash average_models.sh\n"
cd ../../experiments/checkpoints/USPTO50K-aug-untyped_20210423
bash average_models.sh

# somehow get error if running in the same bash script
# printf "Running bash translate.sh\n"
# cd ../../../script/USPTO50K-aug-untyped
# bash translate.sh

# printf "Running score.sh\n"
# bash score.sh