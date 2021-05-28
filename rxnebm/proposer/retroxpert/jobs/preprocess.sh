#!/bin/bash
#SBATCH -n 1                                # number of comands issued in batch script at any times
#SBATCH --cpus-per-task=16                  # number of cores to allocate each task
#SBATCH -N 1                                # Request 1 node
#SBATCH --time=1:00:00

#SBATCH --gres=gpu:0                       # specify GPU count
#SBATCH -p sched_mit_ccoley                 # sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=8000                  # Request 4G of memory per CPU

#SBATCH -o logs/prep_output_%x_%j.txt               # redirect output to output_JOBNAME_JOBID.txt
#SBATCH -e logs/prep_error_%x_%j.txt                # redirect errors to error_JOBNAME_JOBID.txt
#SBATCH -J xpert_prep                           # name of job
#SBATCH --mail-type=ALL               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

#SBATCH --nodelist node1238

flags=$1

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

# run these to make graphfeats in .pkl (one .pkl per reaction, so there are MANY .pkl files)
printf "running processing.py\n"
python preprocessing.py

# no need to run with --extract_pattern, 527 templates with cnt >= 2 already in data/USPTO50K/product_patterns.txt
# subtle bug raised in github/Issues but authors do not provide a solution. If this is run again, a number of templates DIFFERENT
# from 527 will be generated. Hence we highly recommend using our provided product_patterns.txt

if [ flags = '--extract_pattern' ]
then
    # these 2 lines will run only if you are extracting templates on your own dataset
    printf "running extract_semi_template_pattern.py with --extract_pattern\n"
    python extract_semi_template_pattern.py --extract_pattern
fi

printf "running extract_semi_template_pattern.py\n"
python extract_semi_template_pattern.py