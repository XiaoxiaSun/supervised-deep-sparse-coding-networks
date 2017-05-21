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
void __global__ col2im_gpu_kernel(float const * X_col,
        int const num_kernel, int const ksize, const int stride,
        int const H, int const W,
        int const K, int const N, float * X)
{
//     Input:
//     X_col: K*kw*kh x N*H*W  (M x L)
//     Output:
//     X: H x W x K x N
    const int pad_size = ksize / 2;
//     const int K_out = K * ksize * ksize;
    const int H_small = (H+2*pad_size - ksize) / stride + 1;
    const int W_small = (W+2*pad_size - ksize) / stride + 1;
    int P =  ksize * ksize;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < num_kernel;
    i += blockDim.x * gridDim.x){
        //         i = h + H_small * w + (H_small*W_small)*k + (H_small*W_small*K)*n
        
        int h = i % H_small;
        int w = (i / H_small) % H_small;
        int k = (i / (H_small*W_small)) % K;
        int n = i / (H_small*W_small*K);
        
        int h_dense = h * stride - pad_size;
        int w_dense = w * stride - pad_size;
        
//         float val = 0;
        for(int kw = 0; kw <ksize; kw++){
            for(int kh = 0; kh <ksize; kh++){ 
                int h_c = h_dense + kh ;
                int w_c = w_dense + kw;
//                 int channel_idx = kw + kh *ksize;
//                 int idx_in = batch_idx + batch_size*channel_idx + (batch_size*nchannel_in)*w + (batch_size*nchannel_in*width)*h;
                if(h_c>=0 && w_c>=0 && h_c<H && w_c<W){
                    X[h_c + H * w_c + (H*W)*k + (H*W*K)*n]  +=   X_col[kh + ksize*kw + P*k + (P*K)*w + (P*K*W_small)*h + (P*K*W_small*H_small)*n];
                }
                __syncthreads();
            }
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
        //          X_col: M x L (m*nchannel x num_pixel*num_patch)
        //          H: height of output image tensor
        //          W: width of output image tensor
        //          K: number of channel
        //          N: mini batch size
        //          ksize: size of kernel
        //          stride: sampling step
        //
        //     Output:
        //          X: image tensor,  H x W x K x N
{
    /* Declare all variables.*/
    mxGPUArray const *X_col;
    mxGPUArray *X;
    float const *d_X_col;
    float *d_X;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input number/type to MEX file.";
    
    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 256;
    int blocksPerGrid;
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=7) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
//      printf("1");
    X_col = mxGPUCreateFromMxArray(prhs[0]);
    const int H = (const int)mxGetScalar(prhs[1]);
    const int W = (const int)mxGetScalar(prhs[2]);
    const int K = (const int)mxGetScalar(prhs[3]);
    const int N = (const int)mxGetScalar(prhs[4]);
    const int ksize = (const int)mxGetScalar(prhs[5]);
    const int stride = (const int)mxGetScalar(prhs[6]);
    
//     printf("%d, %d, %d, %d, %d\n", H, W, K, N, ksize);
//pad_size" is undefined
//      printf("2");
    mwSize const * size_X_col = mxGPUGetDimensions(X_col);
//  printf("3");
    const int M = (const int)size_X_col[0];
    const int L  = (const int)size_X_col[1];
//     printf("height = %d, width = %d, nchannel_in = %d, num_batch = %d\n", height, width, nchannel_in, num_batch);
//  printf("4");
    if (mxGPUGetClassID(X_col) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, "Input must be single!");
    }
    
//      printf("a");
//     if (L!=H*W*N || M!=ksize*ksize*K) {
//         mexErrMsgIdAndTxt(errId, "Invalid input size!");
//     }
    
//  printf("b");
    d_X_col = (float const *)(mxGPUGetDataReadOnly(X_col));
    
    
    const mwSize size_X[] = {(const mwSize)H, (const mwSize)W, (const mwSize)K, (const mwSize)N};
//     mwSize const* size_X_col = (mwSize const*)(H*W*N*K_out);
//   printf("6");
//     printf("col = %d\n", size_X_col[1]);
    /* Create a GPUArray to hold the result and get its underlying pointer. */
    X   = mxGPUCreateGPUArray(4,
            size_X,
            mxSINGLE_CLASS,
            mxREAL,
            MX_GPU_INITIALIZE_VALUES );
    d_X = (float *)(mxGPUGetData(X));
//      printf("7");
//     printf("1");
     const int pad_size = ksize / 2;
        const int H_small = (H+2*pad_size - ksize) / stride + 1;
    const int W_small = (W+2*pad_size - ksize) / stride + 1;
    const int num_gpu_kernels = (const int)K*H_small*W_small*N;
//      printf("8");
//     const int N = height*width;
//         printf("2");
//     printf("ksize = %d\n", ksize);
    int num = K*H_small*W_small*N;
    blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;
    col2im_gpu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X_col, num_gpu_kernels, ksize, stride, H,
            W, K, N, d_X);
    
    
//         printf("3");
//      printf("9");
    
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(X);
//      printf("10");
    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(X);
    mxGPUDestroyGPUArray(X_col);
//      printf("11\n");
}
