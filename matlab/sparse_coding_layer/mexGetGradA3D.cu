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
void __global__ getC_gpu_kernel(float const* A, float const* X, float const* E, float const* B,
        int const M, int const N, int const P, float* C)
{
        // computes C = A x X + E x B;
        //      Input:
        //          A: M x P
        //          X: N x P
        //          E: M x P
        //          B: N x P
        //     Output:
        //          C: M x N x P
    
//     extern __shared__ float s[];
//     float *sAtA = s;
    const int num_kernel = M*N*P;
    
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
        //         i = m + M * n + (M*N)*p
        int m = i % M;
        int n = (i / M) % N;
        int p = i / (M*N);
        
        C[i] = A[m+M*p] * X[n+N*p] + E[m+M*p] * B[n+N*p];
        
    }
}

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, mxArray const *prhs[])
        // gradA = mexGetGradA3D(A, X, E, B);
        // computes C = A x X + E x B;
        //      Input:
        //          A: M x P
        //          X: N x P
        //
        //          E: M x P
        //          B: N x P
        //     Output:
        //          C: M x N x P
{
    /* Declare all variables.*/
    mxGPUArray const *A, *X, *E, *B;
    mxGPUArray *C;
    float const *d_A, *d_X, *d_E, *d_B;
    float *d_C;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input number/type to MEX file.";
    
    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 256;
    int blocksPerGrid;
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=4) || !(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1]))
    || !(mxIsGPUArray(prhs[2])) || !(mxIsGPUArray(prhs[3]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    A = mxGPUCreateFromMxArray(prhs[0]);
    X = mxGPUCreateFromMxArray(prhs[1]);
    E = mxGPUCreateFromMxArray(prhs[2]);
    B = mxGPUCreateFromMxArray(prhs[3]);
    
    mwSize const * size_A = mxGPUGetDimensions(A);
    mwSize const * size_X= mxGPUGetDimensions(X);
    
    const int M = size_A[0];
    const int P  = size_A[1];
    const int N  = size_X[0];
    //printf("N = %d, Nmax = %d, P = %d\n", N, Nmax, P);
    
    if (mxGPUGetClassID(A) != mxSINGLE_CLASS || mxGPUGetClassID(X) != mxSINGLE_CLASS
            || mxGPUGetClassID(E) != mxSINGLE_CLASS || mxGPUGetClassID(B) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, "Input must be single!");
    }
    
    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    d_X = (float const *)(mxGPUGetDataReadOnly(X));
    d_E = (float const *)(mxGPUGetDataReadOnly(E));
    d_B = (float const *)(mxGPUGetDataReadOnly(B));
    
    const mwSize size_C[] = {M, N, P};
    
    /* Create a GPUArray to hold the result and get its underlying pointer. */
    C   = mxGPUCreateGPUArray(3,
            size_C,
            mxSINGLE_CLASS,
            mxREAL,
            MX_GPU_INITIALIZE_VALUES );
    d_C = (float *)(mxGPUGetData(C));
    
    
    // const int num_gpu_kernels = (const int)Nmax*Nmax*P;
    int num = (int)(mxGPUGetNumberOfElements(C));
    blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;
    getC_gpu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_X, d_E, d_B, M, N, P, d_C);
    
    
//         printf("3");
//      printf("9");
    
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(C);
//      printf("10");
    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(X);
    mxGPUDestroyGPUArray(E);
    mxGPUDestroyGPUArray(B);
    mxGPUDestroyGPUArray(C);
//      printf("11\n");
}
