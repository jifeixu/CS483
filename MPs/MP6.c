// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

__global__ void floatToUnsignedChar(float *inputImage, unsigned char *ucharImage, int width, int height, int channels)
{
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  int ty = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (tx<width && ty<height)
  {
    int ii = blockIdx.z * width * height + ty * width + tx;
    ucharImage[ii] = (unsigned char) ((HISTOGRAM_LENGTH - 1) * inputImage[ii]);
  } 
}
 
__global__ void colortoGrayScaleConversion(unsigned char *ucharImage, unsigned char *grayImage, int width, int height, int channels)
{
  int Col = threadIdx.x + blockIdx.x * blockDim.x;
  int Row = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (Col<width && Row<height)
  {
    int grayOffset = Row * width + Col;
    
    int rgbOffset = grayOffset*channels;
    unsigned char r = ucharImage[rgbOffset];
    unsigned char g = ucharImage[rgbOffset + 1];
    unsigned char b = ucharImage[rgbOffset + 2];
    
    grayImage[grayOffset] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void grayScaleToHistogram(unsigned char *grayImage, unsigned int *histogram, int width, int height, int channels)
{  
/*  __shared__ unsigned int histo_s[HISTOGRAM_LENGTH];
  
  int tIdx = threadIdx.x + threadIdx.y * blockDim.x;
  
  //int tx = threadIdx.x + blockIdx.x * blockDim.x;
  //int ty = threadIdx.y + blockIdx.y * blockDim.y;
  
  //int tIdx = ty * width + tx;
  
  if (tIdx < HISTOGRAM_LENGTH)
  {
    histo_s[tIdx]=0;
  }
  __syncthreads();
  
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  int ty = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (tx<width && ty<height)
  {
    int ii = ty * width + tx;
    atomicAdd(&histo_s[grayImage[ii]],1);
  }
  __syncthreads();
  
  if (tIdx<HISTOGRAM_LENGTH)
  {
    atomicAdd(&(histogram[tIdx]), histo_s[tIdx]);
  }
  */
  
 int i = threadIdx.x + blockIdx.x * blockDim.x;
  
 if (i<width*height)
 {
   unsigned char val = grayImage[i];
   atomicAdd(&(histogram[val]),1);
 } 
  
}

__global__ void histogramToCDF(unsigned int *histogram, float *deviceImageCDF, int width, int height, int channels)
{
  __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
  int x = threadIdx.x;
  cdf[x] = histogram[x];

  // First scan half
  for (unsigned int stride = 1; stride <= HISTOGRAM_LENGTH / 2; stride *= 2) {
    __syncthreads();
    int idx = (x + 1) * 2 * stride - 1;
    if (idx < HISTOGRAM_LENGTH) {
      cdf[idx] += cdf[idx - stride];
    }
  }

  // Second scan half
  for (int stride = HISTOGRAM_LENGTH / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int idx = (x + 1) * 2 * stride - 1;
    if (idx + stride < HISTOGRAM_LENGTH) {
      cdf[idx + stride] += cdf[idx];
    }
  }

  __syncthreads();
  deviceImageCDF[x] = cdf[x] / ((float) (width * height));  

}

__global__ void equalizeHistogramAndApply(unsigned char *deviceInputImageUChar, unsigned char *ucharCorrectImage, float *deviceImageCDF, int width, int height, int channels)
{
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  int ty = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (tx<width && ty<height)
  {
    int ii = blockIdx.z * width * height + ty * width + tx;
    unsigned char val = deviceInputImageUChar[ii];
    float equalized = (HISTOGRAM_LENGTH - 1) * (deviceImageCDF[val] - deviceImageCDF[0]) / (1.0 - deviceImageCDF[0]);
    float clamp = min(max(equalized,0.0), 255.0);
    
    ucharCorrectImage[ii] = (unsigned char) (clamp);
  }
}

__global__ void convertToFloat(unsigned char *ucharCorrectImage, float *outputImage, int width, int height, int channels)
{
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  int ty = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (tx<width && ty<height)
  {
    int ii = blockIdx.z * width * height + ty * width + tx;
    outputImage[ii] = (float) (ucharCorrectImage[ii]/255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  unsigned char *deviceInputImageUChar;
  unsigned char *deviceInputImageUCharGrayScale;

  unsigned int *deviceImageHistogram;
  float *deviceImageCDF;
 
  unsigned char *deviceCorrectImage;
  float *deviceOutputImageData;
 
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  
  hostInputImageData = wbImage_getData(inputImage); 
  hostOutputImageData = wbImage_getData(outputImage); 
  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  // printf("%d, %d, %d\n", imageWidth, imageHeight, imageChannels);
  cudaMalloc((void**) &deviceInputImageData, imageWidth * imageHeight *imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceInputImageUChar, imageWidth * imageHeight *imageChannels * sizeof(unsigned char));
  cudaMalloc((void**) &deviceInputImageUCharGrayScale, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void**) &deviceImageHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
//  cudaMemset((void *) deviceImageHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset(deviceImageHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void**) &deviceImageCDF, HISTOGRAM_LENGTH * sizeof(float));
  
  cudaMalloc((void**) &deviceCorrectImage, imageWidth * imageHeight *imageChannels * sizeof(unsigned char));
  cudaMalloc((void**) &deviceOutputImageData, imageWidth * imageHeight *imageChannels * sizeof(float));
  
  // kernel call: float to unsigned char
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight *imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dim3 dimBlock(32,32,1);
  floatToUnsignedChar<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceInputImageUChar, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  printf("Out of float to char kernel \n");

//  debugging
  unsigned char *hostInputImageUChar;
  hostInputImageUChar = (unsigned char *) malloc(imageWidth * imageHeight *imageChannels);
  cudaMemcpy(hostInputImageUChar, deviceInputImageUChar, imageWidth * imageHeight *imageChannels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  printf("hostInputImageData %f\n", hostInputImageData[0]);
  printf("hostInputImageUChar %u\n", hostInputImageUChar[0]);

  
  // kernel call: grayscale
  colortoGrayScaleConversion<<<dimGrid,dimBlock>>>(deviceInputImageUChar,deviceInputImageUCharGrayScale, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  printf("Out of color to gray scale conversion kernel \n");
  
  //  debugging
  unsigned char *hostInputImageUCharGrayScale;
  hostInputImageUCharGrayScale = (unsigned char *) malloc(imageWidth * imageHeight);
  cudaMemcpy(hostInputImageUCharGrayScale, deviceInputImageUCharGrayScale, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  printf("hostInputImageUCharGrayScale %u\n", hostInputImageUCharGrayScale[0]);
  
  // kernel call: grayscale to histogram
  // For kernel with privatization
//  dimGrid = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
//  dimBlock = dim3(32,32,1);
//  For no privatization kernel
  dimGrid = dim3(ceil((imageWidth*imageHeight)/32.0),1,1);
  dimBlock = dim3(32,1,1);
  grayScaleToHistogram<<<dimGrid, dimBlock>>>(deviceInputImageUCharGrayScale, deviceImageHistogram, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  printf("Out of histogram kernel \n");
  
  //  debugging
  unsigned int *hostImageHistogram;
  hostImageHistogram = (unsigned int *) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
  memset(hostImageHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
//  cudaMalloc((void**) &deviceImageHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
//  cudaMemset((void *) deviceImageHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
//  cudaMemset(deviceImageHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemcpy(hostImageHistogram, deviceImageHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("hostImageHistogram %u\n", hostImageHistogram[0]);
  
  for (int i=0; i<256; i++)
  {
    printf("%u, ", hostImageHistogram[i]);
  }
  printf("\n");
  
  // kernel call: histogram to cdf
  dimGrid = dim3(1,1,1);
  dimBlock= dim3(HISTOGRAM_LENGTH,1,1);
  histogramToCDF<<<dimGrid, dimBlock>>>(deviceImageHistogram, deviceImageCDF, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  printf("Out of histogram to CDF kernel \n");
  
  //  debugging
  float *hostImageCDF;
  hostImageCDF = (float *) malloc(HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(hostImageCDF, deviceImageCDF, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
  printf("hostImageCDF %f\n", hostImageCDF[0]);
  for (int i=0; i<256; i++)
  {
    printf("%f, ", hostImageCDF[i]);
  }
  printf("\n");
  
  // kernel call: histogram equalization
  dimGrid = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock= dim3(32,32,1);
  equalizeHistogramAndApply<<<dimGrid, dimBlock>>>(deviceInputImageUChar,deviceCorrectImage, deviceImageCDF, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  printf("Out of equalize histogram and apply equalization \n");
  
  //  debugging
  unsigned char *hostCorrectImage;
  hostCorrectImage = (unsigned char *) malloc(imageWidth * imageHeight * imageChannels);
  cudaMemcpy(hostCorrectImage, deviceCorrectImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  printf("hostCorrectImage %u\n", hostCorrectImage[0]);
  for (int i=0; i<imageWidth*imageHeight; i++)
  {
    printf("%u, ", hostCorrectImage[i]);
  }
  printf("\n");
  
  // kernel call: convert to float
  convertToFloat<<<dimGrid,dimBlock>>>(deviceCorrectImage, deviceOutputImageData, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  printf("Out of convert to float kernel \n");
  
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight *imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  
  printf("hostOutputImageData %f\n", hostOutputImageData[0]);
  
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);

  return 0;
}
