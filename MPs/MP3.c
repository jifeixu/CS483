#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 8

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];  
  
  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int tx = threadIdx.x; 
  int ty = threadIdx.y;
  
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  
  float Cvalue=0;
  
  for (int ph =0 ; ph < (TILE_WIDTH + numAColumns - 1)/TILE_WIDTH; ++ph)
  {
     if ((Row<numARows) && (ph*TILE_WIDTH+tx)<numAColumns)
       Mds[ty][tx] = A[Row*numAColumns+ph*TILE_WIDTH + tx];
     else
       Mds[ty][tx] = 0.0 ; 
    
     if ((ph*TILE_WIDTH+ty)<numBRows && Col<numBColumns)
         Nds[ty][tx] = B[(ph*TILE_WIDTH + ty)*numBColumns + Col];
     else
        Nds[ty][tx] = 0.0 ; 
    
     __syncthreads();
     for (int k = 0; k < TILE_WIDTH; ++k)
     {
       Cvalue += Mds[ty][k] * Nds[k][tx];
     }
    __syncthreads();
  }
  if ((Row<numCRows) && (Col<numCColumns)) 
  {
    C[((by * blockDim.y + ty)*numCColumns) + (bx * blockDim.x)+ tx] = Cvalue;
  }
}

#define BLOCK_SIZE 8

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  int size_C = numCRows * numCColumns;
  int mem_size_C = sizeof(float) * size_C;
  hostC = (float *)malloc(mem_size_C);

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  
  int size_A = numARows * numAColumns;
  int mem_size_A = sizeof(float) * size_A;
  
  int size_B = numBRows * numBColumns;
  int mem_size_B = sizeof(float) * size_B;
  
  cudaMalloc((void **) &deviceA, mem_size_A);
  cudaMalloc((void **) &deviceB, mem_size_B);
  cudaMalloc((void **) &deviceC, mem_size_C);
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  cudaMemcpy(deviceA, hostA, mem_size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, mem_size_B, cudaMemcpyHostToDevice);
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
  dim3 dimGrid((numBColumns + BLOCK_SIZE -1)/ BLOCK_SIZE,(numARows + BLOCK_SIZE -1)/ BLOCK_SIZE,1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid,dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, mem_size_C, cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
