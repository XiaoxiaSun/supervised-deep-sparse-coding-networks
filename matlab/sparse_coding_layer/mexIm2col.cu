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
void __global__ im2col_gpu_kernel(float const * X,
        int const num_kernel, int const ksize, const int stride, int const H, int const W,
        int const K, int const N, float * X_col)
{
//     Input:
//     X: H x W x K x N
//     Output:
//     X_col: K*kw*kh x N*H_out*W_out
    const int pad_size = ksize / 2;
//     const int K_out = K * ksize * ksize;
    const int H_out = (H+2*pad_size - ksize) / stride + 1;
    const int W_out = (W+2*pad_size - ksize) / stride + 1;
    const int P =  ksize * ksize;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < num_kernel;
        i += blockDim.x * gridDim.x){
//         i = h_out + H_out*w_out + H_out*W_out*k + H_out*W_out*K_out*n
        int h_out = i % H_out;
        int w_out = (i / H_out) % H_out;
        int k = (i / (H_out*W_out)) % K;
        int n = i / (H_out*W_out*K);
        
        
        int h_in = h_out * stride - pad_size;
        int w_in = w_out * stride - pad_size;

//         X_col[i] = X[i];
//          X_col: N*H*W x K*kw*kh
            for(int kw = 0; kw <ksize; kw++){
                for(int kh = 0; kh <ksize; kh++){
                int h_c = h_in + kh;
                int w_c = w_in + kw;
                    X_col[kh + ksize*kw + P*k + (P*K)*w_out + (P*K*W_out)*h_out + (P*K*W_out*H_out)*n] = (h_c>=0 && w_c>=0 && h_c<H && w_c<W) ?
                                                                                                    X[h_c + H * w_c + (H*W)*k + (H*W*K)*n] : 0;
                
                
//                     X_col[kh + ksize*kw + P*k + (P*K)*w_out + (P*K*W_out)*h_out + (P*K*W_out*H_out)*n] = (h_c>=0 && w_c>=0 && h_c<H && w_c<W) ?
//                                                                                                     X[h_c + H * w_c + (H*W)*k + (H*W*K)*n] : 0;
                    
                   
            }
        }
        
    }
}

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, mxArray const *prhs[])
        // Alpha = mexIm2col(X, kernel_size);
        // Input:
        //     X: height x width x nchannel_in x num_batch
        //     kernel_size: size of kernel
        //     stride: sampling rate  
//   Output:
//                  X_col: kerne_size^2*nchannel x H*W*num_batch
{
    /* Declare all variables.*/
    mxGPUArray const *X;
    mxGPUArray *X_col;
    float const *d_X;
    float *d_X_col;
    int ksize;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    
    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 256;
    int blocksPerGrid;
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=3) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    X = mxGPUCreateFromMxArray(prhs[0]);
    ksize = (int)mxGetScalar(prhs[1]);
    const int stride = (int)mxGetScalar(prhs[2]);
    int pad_size = ksize / 2;
    
    mwSize const * size_X = mxGPUGetDimensions(X);
    mwSize const ndim_X = mxGPUGetNumberOfDimensions (X);

    int H = (int)size_X[0];
    int W = (int)size_X[1];
    int K = (int)size_X[2];
    int N = (int)size_X[3];
    if(ndim_X==3){
        N = 1;
    }
//     printf("height = %d, width = %d, nchannel_in = %d, num_batch = %d\n", height, width, nchannel_in, num_batch);

    if (mxGPUGetClassID(X) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, "Input must be single!");
    }
    
    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_X = (float const *)(mxGPUGetDataReadOnly(X));
    
    int K_out = K * ksize * ksize;
    const int H_out = (H+2*pad_size - ksize) / stride + 1;
    const int W_out = (W+2*pad_size - ksize) / stride + 1;
    mwSize size_X_col[] = { (mwSize)K_out, (mwSize)H_out*W_out*N};
//     mwSize const* size_X_col = (mwSize const*)(H*W*N*K_out);
 
//     printf("col = %d\n", size_X_col[1]);
    /* Create a GPUArray to hold the result and get its underlying pointer. */
    X_col   = mxGPUCreateGPUArray(2,
            size_X_col,
            mxSINGLE_CLASS,
            mxREAL,
            MX_GPU_DO_NOT_INITIALIZE );
    d_X_col = (float *)(mxGPUGetData(X_col));
    
//     printf("1");
    const int num_gpu_kernels = (const int)K*H_out*W_out*N;
//     const int N = height*width;
//         printf("2");
    blocksPerGrid = (num_gpu_kernels + threadsPerBlock - 1) / threadsPerBlock;
    im2col_gpu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, num_gpu_kernels, ksize, stride, H,
            W, K, N, d_X_col);
//         printf("3");
    
    
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(X_col);
//        printf("4");
    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(X);
//        printf("5");
    mxGPUDestroyGPUArray(X_col);
//        printf("6");
}
