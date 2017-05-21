/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
// #include "math.h"

// #define CUDA_KERNEL_LOOP(i, n)
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x;
//            i < (n);
//            i += blockDim.x * gridDim.x)

/*
 * Device code
 */
void __global__ getAtAact_gpu_kernel(float const* AtA, float const* actIdx, int const N, int const P, int const Nmax, float* AtAact)
{
    //      Input:
    //          AtA: N x N
    //          actIdx: Nmax x P, indicator matrix
    //     Output:
    //          AtAact: image tensor,  Nmax x Nmax x P
    
//     extern __shared__ float s[];
//     float *sAtA = s;
    const int num_kernel = Nmax*Nmax*P;
    
//     // copy AtA to shared memory
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x;
//     i < N*N;
//     i += blockDim.x * gridDim.x){
//         sAtA[i] = AtA[i];
//         __syncthreads();
//     }
    
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < num_kernel;
    i += blockDim.x * gridDim.x){
        //         i = m + Nmax * n + (Nmax*Nmax)*p
        int m = i % Nmax;
        int n = (i / Nmax) % Nmax;
        int p = i / (Nmax*Nmax);
        
        // assigning active AtA from sAtA
        int mact = actIdx[m+Nmax*p] - 1; // matlab indices starts from 1
        int nact = actIdx[n+Nmax*p] - 1;
        //AtAact[i] = Nmax;
        if(nact!=-1 && mact!=-1){            
         AtAact[m+Nmax*n+(Nmax*Nmax)*p] = AtA[mact + nact*N];
//             AtAact[p] = p;
        }
        
    }
}

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, mxArray const *prhs[])
        // img = mexCol2im(X_col, H, W, K, N, ksize);
        //
        //      Input:
        //          AtA: N x N
        //          actIdx: Nmax x P, indicator matrix
        //     Output:
        //          AtAact: image tensor,  Nmax x Nmax x P
{
    /* Declare all variables.*/
    mxGPUArray const *AtA;
    mxGPUArray const *actIdx;
    mxGPUArray *AtAact;
    float const *d_AtA, *d_actIdx;
    float *d_AtAact;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input number/type to MEX file.";
    
    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 256;
    int blocksPerGrid;
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=2) || !(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
//      printf("1");
    AtA = mxGPUCreateFromMxArray(prhs[0]);
    actIdx = mxGPUCreateFromMxArray(prhs[1]);
    
    mwSize const * size_AtA = mxGPUGetDimensions(AtA);
    mwSize const * size_actIdx = mxGPUGetDimensions(actIdx);
    
    const int N = size_AtA[0];
    const int Nmax  = size_actIdx[0];
    const int P  = size_actIdx[1];
    //printf("N = %d, Nmax = %d, P = %d\n", N, Nmax, P);
    
    if (mxGPUGetClassID(AtA) != mxSINGLE_CLASS || mxGPUGetClassID(actIdx) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, "Input must be single!");
    }
    
    d_AtA = (float const *)(mxGPUGetDataReadOnly(AtA));
    d_actIdx = (float const *)(mxGPUGetDataReadOnly(actIdx));
    
    const mwSize size_AtAact[] = {Nmax, Nmax, P};
    
    /* Create a GPUArray to hold the result and get its underlying pointer. */
    AtAact   = mxGPUCreateGPUArray(3,
            size_AtAact,
            mxSINGLE_CLASS,
            mxREAL,
            MX_GPU_INITIALIZE_VALUES );
    d_AtAact = (float *)(mxGPUGetData(AtAact));
    
    
   // const int num_gpu_kernels = (const int)Nmax*Nmax*P;
    int num = (int)(mxGPUGetNumberOfElements(AtAact));
    blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;
    getAtAact_gpu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_AtA, d_actIdx, N, P, Nmax, d_AtAact);
    
    
//         printf("3");
//      printf("9");
    
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(AtAact);
//      printf("10");
    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(AtA);
    mxGPUDestroyGPUArray(actIdx);
    mxGPUDestroyGPUArray(AtAact);
//      printf("11\n");
}
