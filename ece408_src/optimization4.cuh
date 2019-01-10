#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 1024
#define TILE_WIDTH 24

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void matrixMultiplyShared(float *Kernel, float *X, float *Y, int M, int C, int H, int W, int K, int H_out, int W_out) 
{
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  int numAColumns = C*K*K;
  int numCColumns = H_out*W_out;
  X += blockIdx.z*C*H*W;
  Y += blockIdx.z*M*numCColumns;
  
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int rowIx = blockIdx.y * TILE_WIDTH + ty;
  int colIx = blockIdx.x * TILE_WIDTH + tx;

  float result = 0;
  int q, p, c, w, h;

  for (int tileIx = 0; tileIx < ceil(1.0*numAColumns/TILE_WIDTH);  ++tileIx) {

    int temp       = tileIx*TILE_WIDTH;
    int matrix_col = temp+tx;
    if (rowIx < M && matrix_col < numAColumns)
      tileA[ty][tx] = Kernel[rowIx*numAColumns+matrix_col];  
    else
      tileA[ty][tx] = 0;

    int matrix_row = temp+ty;
    if (colIx < numCColumns && matrix_row < numAColumns) {
      q = matrix_row % K;
      matrix_row /= K;
      p = matrix_row % K;
      c = matrix_row / K;
      w = colIx % W_out;
      h = colIx / W_out;
      tileB[ty][tx] = X[c * (H * W) + (h+p) * (W) + w+q];
    }
    else 
      tileB[ty][tx] = 0;

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++)
       result += tileA[ty][k]*tileB[k][tx];

    __syncthreads();   
  }
  
  if ((rowIx < M) && (colIx < numCColumns)) {
    Y[rowIx*numCColumns+colIx] = result;
  }
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = k.shape_[3];
    const int H_out = H-K+1;
    const int W_out = W-K+1;
 
    int blockDimX = TILE_WIDTH, blockDimY = TILE_WIDTH;
    int gridDimY = ceil(1.0*M/blockDimY), gridDimX = ceil(1.0*H_out*W_out/blockDimX);
    dim3 gridDim (gridDimX, gridDimY, B), blockDim (blockDimX, blockDimY, 1);
    matrixMultiplyShared<<<gridDim, blockDim>>>(k.dptr_, x.dptr_, y.dptr_, M, C, H, W, K, H_out, W_out);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
