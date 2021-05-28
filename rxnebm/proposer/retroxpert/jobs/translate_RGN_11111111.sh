#!/bin/bash
#SBATCH -n 1                                # number of comands issued in batch script at any times
#SBATCH --cpus-per-task=8                  # number of cores to allocate each task
#SBATCH -N 1                                # Request 1 node
#SBATCH --time=2:00:00

#SBATCH --gres=gpu:1                      # specify GPU count
#SBATCH -p sched_mit_ccoley                 # sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=8000                  # Request 4G of memory per CPU

#SBATCH -o logs/translateRGN_output_%x_%j.txt               # redirect output to output_JOBNAME_JOBID.txt
#SBATCH -e logs/translateRGN_error_%x_%j.txt                # redirect errors to error_JOBNAME_JOBID.txt
#SBATCH -J translateRGN_11111111                         # name of job
#SBATCH --mail-type=ALL               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

printf "Running bash translate.sh\n"
cd OpenNMT-py/script/USPTO50K-aug-untyped
bash translate.sh 11111111

printf "Running score.sh\n"
bash score.sh 11111111