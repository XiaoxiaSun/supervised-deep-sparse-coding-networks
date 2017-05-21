close all; clc; clear;

disp('GPU and CUDA required to compile and test the code.')
disp('The code is tested on Linux 14.01 with 3 Nvidia Titan X (Pascal) GPU / 4 Nvidia Tesla P40 GPU')

addpath_sparse_coding_layer;
vl_compilenn('enableGpu', true);

cd matlab/sparse_coding_layer;

mexcuda mexCol2im.cu -lc -lstdc++;
mexcuda mexIm2col.cu -lc -lstdc++;
mexcuda mexGetActiveAtA.cu -lc -lstdc++;
mexcuda mexGetGradA3D.cu -lc -lstdc++;

cd ..
cd ..
