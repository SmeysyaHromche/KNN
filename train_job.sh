#!/bin/bash
#PBS -N knn_ocr_train
#PBS -l select=1:ncpus=1:mem=32gb:ngpus=1:gpu_mem=30gb:scratch_local=128gb
#PBS -l walltime=24:20:00
#PBS -q gpu

export TMPDIR=$SCRATCHDIR

DATADIR=/storage/brno2/home/$USER/KNN

# Copy the insides of DATADIR
cp -r $DATADIR/* $SCRATCHDIR
cd $SCRATCHDIR

module load python/3.11.11-gcc-10.2.1-555dlyc

pip3.11 install -e .

python3 -m src.learn.train

# Clean the scratch
clean_scratch
