#!/bin/bash
#SBATCH -n 1                                # number of comands issued in batch script at any times
#SBATCH --cpus-per-task=1                 # number of cores to allocate each task
#SBATCH -N 1                                # Request 1 node
#SBATCH --time=0:10:00

#SBATCH --gres=gpu:0                       # specify GPU count
#SBATCH -p sched_mit_ccoley                 # sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=4000                  # Request 4G of memory per CPU

#SBATCH -o logs/chain_output_%x_%j.txt               # redirect output to output_JOBNAME_JOBID.txt
#SBATCH -e logs/chain_error_%x_%j.txt                # redirect errors to error_JOBNAME_JOBID.txt
#SBATCH -J xpert_chain_11111111                           # name of job
#SBATCH --mail-type=FAIL               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

#SBATCH --nodelist node1238

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

## ! /bin/bash
# from https://hpc.nih.gov/docs/job_dependencies.html

# don't do this (for now), make a fresh copy better
# cp /pool001/linmin001/retroxpert_backuprepo/data . -r

seed=11111111

# run this
jid0=$(sbatch --parsable jobs/prepare_data.sh)
jid1=$(sbatch --parsable --dependency=afterok:$jid0 jobs/train_EGAT_${seed}.sh)
jid2=$(sbatch --parsable --dependency=afterok:$jid1 jobs/train_RGN_${seed}.sh)
jid3=$(sbatch --parsable --dependency=afterok:$jid2 jobs/translate_RGN_${seed}.sh)
jid4=$(sbatch --parsable --dependency=afterok:$jid3 jobs/propose_${seed}.sh)

# # a single job can depend on multiple jobs
# jid4=$(sbatch  --dependency=afterany:$jid2:$jid3 job4.sh)

# # a single job can depend on all jobs by the same user with the same name
# jid7=$(sbatch --dependency=afterany:$jid6 --job-name=dtest job7.sh)
# jid8=$(sbatch --dependency=afterany:$jid6 --job-name=dtest job8.sh)
# sbatch --dependency=singleton --job-name=dtest job9.sh

# show dependencies in squeue output:
# squeue -u $USER