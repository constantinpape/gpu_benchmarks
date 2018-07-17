#! /bin/bash

PREFIX=$1

cd  ~/Work/software/conda/miniconda3/envs/inferno/lib/python3.6/site-packages
ln -s $PREFIX/Work/my_projects/inferno/inferno inferno
ln -s $PREFIX/Work/my_projects/neurofire/neurofire neurofire
ln -s $PREFIX/Work/my_projects/neuro-skunkworks/skunkworks skunkworks
ln -s $PREFIX/Work/my_projects/simpleference/simpleference simpleference
ln -s $PREFIX/Work/software/bld/affogato/python/affogato affogato
ln -s $PREFIX/Work/software/bld/z5/python/z5py z5py
