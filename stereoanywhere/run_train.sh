#!/bin/bash

# Enable your conda/venv environment
# source ...
# conda activate ...

source "$CONDA_PREFIX/../../etc/profile.d/conda.sh"
conda activate newcvlab

#Set the path to the datasets
PREPATH="/mnt/massa1/datasets"

#Set the path of the initial stereo model (RAFT-Stereo)
STEREOMODEL_PATH="/mnt/massa1/luca/projects/vpp-vs-guided/weights/raftstereo-realsceneflow.pth"

#Set the path of the monocular model (DAv2 Large)
MONOMODEL_PATH="/home/luca/Desktop/projects/Depth-Anything-V2/weights/depth_anything_v2_vitl.pth"

DATASET="sceneflow"
DATAPATH="${PREPATH}/FlyingThings3D_subset/train/;${PREPATH}/Monkaa/;${PREPATH}/Driving/"

DATASETVAL="middlebury"
DATAPATHVAL="${PREPATH}/MiddEval3/trainingH"

SEED=42
SAVEPATH="log/"
INITSTEP=0
EPOCHS=3
SAVESTEP=1
PLOTSTEP=500
IMAGESTEP=1000
VALSTEP=1000
IMAGE_SIZE="320 640"
BATCH=2
LR_RATE=0.0001
MAXDISP=700
ISCALE=1
OSCALE=1

mkdir -p $SAVEPATH

python train.py --dataset $DATASET --datapath $DATAPATH --datasetval $DATASETVAL --datapathval $DATAPATHVAL  \
 --savemodel $SAVEPATH --savestep $SAVESTEP --plotstep $PLOTSTEP --imagestep $IMAGESTEP --valstep $VALSTEP \
 --epochs $EPOCHS --image_size $IMAGE_SIZE --batch $BATCH --lr $LR_RATE  \
 --model stereoanywhere  --initstep $INITSTEP --iscale $ISCALE --oscale $OSCALE\
 --loadmonomodel $MONOMODEL_PATH --loadmodel $STEREOMODEL_PATH \
 --seed $SEED --debug_grad --do_validation --numworkers 8 --freeze_for_finetuning --things_to_freeze fnet \
 --vol_n_masks 8 --volume_channels 8 --vol_aug_n_masks 8 --n_additional_hourglass 0 --use_aggregate_mono_vol \
 --use_border_mask --use_normal_loss_on_coarse --iters 12 \
 --volume_corruption_prob 0.3 --gt_mono_prob 0.3 --preload_mono

