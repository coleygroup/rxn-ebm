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
#SBATCH -J EGAT_orig50K                            # name of job
#SBATCH --mail-type=ALL               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

python3 preprocessing.py

# extract first
python extract_semi_template_pattern.py --extract_pattern

python3 extract_semi_template_pattern.py

python3 train.py

python3 train.py --test_only --load

# Namespace(batch_size=32, dataset='USPTO50K', epochs=80, exp_name='USPTO50K_untyped', gat_layers=3, heads=4, hidden_dim=128, in_dim=693, load=False, logdir='logs', lr=0.0005, seed=123, test_on_train=False, test_only=False, typed=False, use_cpu=False, valid_only=False)
# Counter({1: 3473, 0: 1412, 2: 102, 9: 1, 17: 1})
# Counter({1: 27678, 0: 11185, 2: 838, 3: 4, 4: 4, 10: 2, 7: 1, 13: 1})
# Train Loss: 5.13495
# Train Bond Disconnection Acc: 0.29728
# Train Loss: 3.89347
# Train Bond Disconnection Acc: 0.42150
# Train Loss: 3.61221
# Train Bond Disconnection Acc: 0.45685
# Train Loss: 3.47201
# Train Bond Disconnection Acc: 0.47037
# Train Loss: 3.36373
# Train Bond Disconnection Acc: 0.49029
# Bond disconnection number prediction acc: 0.835638
# Loss:  3.339599835703
# Bond disconnection acc (without auxiliary task): 0.557025
# Train Loss: 3.27819
# Train Bond Disconnection Acc: 0.50190
# Train Loss: 3.18969
# Train Bond Disconnection Acc: 0.51351
# Train Loss: 3.12601
# Train Bond Disconnection Acc: 0.52192
# Train Loss: 3.07477
# Train Bond Disconnection Acc: 0.52968
# Train Loss: 3.03439
# Train Bond Disconnection Acc: 0.53290
# Bond disconnection number prediction acc: 0.856284
# Loss:  3.098413879920591
# Bond disconnection acc (without auxiliary task): 0.567849
# Train Loss: 2.98773
# Train Bond Disconnection Acc: 0.53836
# Train Loss: 2.94512
# Train Bond Disconnection Acc: 0.54619
# Train Loss: 2.91265
# Train Bond Disconnection Acc: 0.55221
# Train Loss: 2.87180
# Train Bond Disconnection Acc: 0.55695
# Train Loss: 2.83793
# Train Bond Disconnection Acc: 0.56206
# Bond disconnection number prediction acc: 0.859491
# Loss:  3.1996192771558367
# Bond disconnection acc (without auxiliary task): 0.563239
# Train Loss: 2.81718
# Train Bond Disconnection Acc: 0.56153
# Train Loss: 2.78891
# Train Bond Disconnection Acc: 0.56871
# Train Loss: 2.76699
# Train Bond Disconnection Acc: 0.56727
# Train Loss: 2.73664
# Train Bond Disconnection Acc: 0.57024
# Train Loss: 2.72174
# Train Bond Disconnection Acc: 0.57231
# Bond disconnection number prediction acc: 0.867108
# Loss:  3.248004637301673
# Bond disconnection acc (without auxiliary task): 0.540188
# Train Loss: 2.43701
# Train Bond Disconnection Acc: 0.62101
# Train Loss: 2.34650
# Train Bond Disconnection Acc: 0.63566
# Train Loss: 2.29827
# Train Bond Disconnection Acc: 0.64153
# Train Loss: 2.26853
# Train Bond Disconnection Acc: 0.64457
# Train Loss: 2.23522
# Train Bond Disconnection Acc: 0.64865
# Bond disconnection number prediction acc: 0.871918
# Loss:  3.078067910291313
# Bond disconnection acc (without auxiliary task): 0.626779
# Train Loss: 2.20977
# Train Bond Disconnection Acc: 0.65205
# Train Loss: 2.18156
# Train Bond Disconnection Acc: 0.65699
# Train Loss: 2.15333
# Train Bond Disconnection Acc: 0.65943
# Train Loss: 2.12826
# Train Bond Disconnection Acc: 0.66381
# Train Loss: 2.10827
# Train Bond Disconnection Acc: 0.66512
# Bond disconnection number prediction acc: 0.871317
# Loss:  3.320377792569434
# Bond disconnection acc (without auxiliary task): 0.631189
# Train Loss: 2.08532
# Train Bond Disconnection Acc: 0.66799
# Train Loss: 2.06418
# Train Bond Disconnection Acc: 0.66751
# Train Loss: 2.04411
# Train Bond Disconnection Acc: 0.67331
# Train Loss: 2.01833
# Train Bond Disconnection Acc: 0.67781
# Train Loss: 1.99979
# Train Bond Disconnection Acc: 0.67950
# Bond disconnection number prediction acc: 0.872920
# Loss:  3.333156497761873
# Bond disconnection acc (without auxiliary task): 0.611946
# Train Loss: 1.98813
# Train Bond Disconnection Acc: 0.67920
# Train Loss: 1.96330
# Train Bond Disconnection Acc: 0.68378
# Train Loss: 1.94581
# Train Bond Disconnection Acc: 0.68678
# Train Loss: 1.92471
# Train Bond Disconnection Acc: 0.68877
# Train Loss: 1.91239
# Train Bond Disconnection Acc: 0.69068
# Bond disconnection number prediction acc: 0.872319
# Loss:  3.4875398545256595
# Bond disconnection acc (without auxiliary task): 0.627180
# Train Loss: 1.78826
# Train Bond Disconnection Acc: 0.71357
# Train Loss: 1.75318
# Train Bond Disconnection Acc: 0.71919
# Train Loss: 1.73641
# Train Bond Disconnection Acc: 0.72163
# Train Loss: 1.72655
# Train Bond Disconnection Acc: 0.72185
# Train Loss: 1.71424
# Train Bond Disconnection Acc: 0.72500
# Bond disconnection number prediction acc: 0.872520
# Loss:  3.873729064293007
# Bond disconnection acc (without auxiliary task): 0.622770
# Train Loss: 1.70710
# Train Bond Disconnection Acc: 0.72410
# Train Loss: 1.69806
# Train Bond Disconnection Acc: 0.72563
# Train Loss: 1.68906
# Train Bond Disconnection Acc: 0.72681
# Train Loss: 1.67883
# Train Bond Disconnection Acc: 0.72805
# Train Loss: 1.67350
# Train Bond Disconnection Acc: 0.72948
# Bond disconnection number prediction acc: 0.874724
# Loss:  4.1262585916555485
# Bond disconnection acc (without auxiliary task): 0.625977
# Train Loss: 1.66592
# Train Bond Disconnection Acc: 0.73052
# Train Loss: 1.65854
# Train Bond Disconnection Acc: 0.73137
# Train Loss: 1.65143
# Train Bond Disconnection Acc: 0.73291
# Train Loss: 1.64213
# Train Bond Disconnection Acc: 0.73354
# Train Loss: 1.63708
# Train Bond Disconnection Acc: 0.73462
# Bond disconnection number prediction acc: 0.868511
# Loss:  4.220614563465405
# Bond disconnection acc (without auxiliary task): 0.621567
# Train Loss: 1.63028
# Train Bond Disconnection Acc: 0.73512
# Train Loss: 1.62351
# Train Bond Disconnection Acc: 0.73505
# Train Loss: 1.61912
# Train Bond Disconnection Acc: 0.73626
# Train Loss: 1.61052
# Train Bond Disconnection Acc: 0.73986
# Train Loss: 1.60311
# Train Bond Disconnection Acc: 0.73948
# Bond disconnection number prediction acc: 0.873321
# Loss:  4.466169310275575
# Bond disconnection acc (without auxiliary task): 0.629786
# Train Loss: 1.56795
# Train Bond Disconnection Acc: 0.74406
# Train Loss: 1.56105
# Train Bond Disconnection Acc: 0.74769
# Train Loss: 1.56067
# Train Bond Disconnection Acc: 0.74681
# Train Loss: 1.55899
# Train Bond Disconnection Acc: 0.74671
# Train Loss: 1.55878
# Train Bond Disconnection Acc: 0.74653
# Bond disconnection number prediction acc: 0.874524
# Loss:  4.478422983444437
# Bond disconnection acc (without auxiliary task): 0.622169
# Train Loss: 1.55562
# Train Bond Disconnection Acc: 0.74688
# Train Loss: 1.55268
# Train Bond Disconnection Acc: 0.74900
# Train Loss: 1.55204
# Train Bond Disconnection Acc: 0.74782
# Train Loss: 1.54853
# Train Bond Disconnection Acc: 0.74928
# Train Loss: 1.54787
# Train Bond Disconnection Acc: 0.74789
# Bond disconnection number prediction acc: 0.875727
# Loss:  4.553184437402911
# Bond disconnection acc (without auxiliary task): 0.622570
# Train Loss: 1.54591
# Train Bond Disconnection Acc: 0.74842
# Train Loss: 1.54544
# Train Bond Disconnection Acc: 0.74902
# Train Loss: 1.54501
# Train Bond Disconnection Acc: 0.74865
# Train Loss: 1.54126
# Train Bond Disconnection Acc: 0.74996
# Train Loss: 1.54186
# Train Bond Disconnection Acc: 0.74885
# Bond disconnection number prediction acc: 0.874123
# Loss:  4.600377654570524
# Bond disconnection acc (without auxiliary task): 0.620565
# Train Loss: 1.53837
# Train Bond Disconnection Acc: 0.75081
# Train Loss: 1.53524
# Train Bond Disconnection Acc: 0.75018
# Train Loss: 1.53541
# Train Bond Disconnection Acc: 0.74925
# Train Loss: 1.53417
# Train Bond Disconnection Acc: 0.74940
# Train Loss: 1.53184
# Train Bond Disconnection Acc: 0.75139
# Bond disconnection number prediction acc: 0.874324
# Loss:  4.65640882720305
# Bond disconnection acc (without auxiliary task): 0.622369

# pred_true_list size: 5005
# Bond disconnection number prediction acc: 0.872727
# Loss:  4.9179023597862095
# Bond disconnection acc (without auxiliary task): 0.597403

python3 train.py --test_on_train --load

# Namespace(batch_size=32, dataset='USPTO50K', epochs=80, exp_name='USPTO50K_untyped', gat_layers=3, heads=4, hidden_dim=128, in_dim=693, load=True, logdir='logs', lr=0.0005, seed=123, test_on_train=True, test_only=False, typed=False, use_cpu=False, valid_only=False)
# Counter({1: 3473, 0: 1412, 2: 102, 9: 1, 17: 1})
# Counter({1: 27678, 0: 11185, 2: 838, 3: 4, 4: 4, 10: 2, 7: 1, 13: 1})
# pred_true_list size: 39713
# Bond disconnection number prediction acc: 0.913353
# Loss:  1.460005446880132
# Bond disconnection acc (without auxiliary task): 0.788608

python3 prepare_data.py

python3 prepare_test_prediction.py

python3 prepare_train_error_aug.py

# Namespace(dataset='USPTO50K', output_suffix='-aug-untyped', typed='untyped')
# guided bond_disconnection prediction cnt and acc: 32696 0.823307229370735
# bond_disconnection len: 39713
# augmentation data size: 7017
# save src_train_aug_err: OpenNMT-py/data/USPTO50K-aug-untyped/src-train-aug-err.txt
# save tgt_train_aug_err: OpenNMT-py/data/USPTO50K-aug-untyped/tgt-train-aug-err.txt
