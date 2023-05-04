#include <stdio.h>
#include <stdint.h>
#define FILTER_WIDTH 3

__constant__ float dc_filter_X[FILTER_WIDTH * FILTER_WIDTH];
__constant__ float dc_filter_Y[FILTER_WIDTH * FILTER_WIDTH];

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};


void readPnm(char * fileName, 
	int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
	char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}


void writeTxt(int * matrix, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			fprintf(f, "%i  ", matrix[j + i*width]);
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

__global__ void convertRgb2GrayKernel(uint8_t * inPixels, int width, int height, 
	uint8_t * outPixels)
{
	// TODO
	// Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	if(x < width && y < height)
	{
		int i = y * width + x;
		uint8_t red = inPixels[3 * i];
		uint8_t green = inPixels[3 * i + 1];
		uint8_t blue = inPixels[3 * i + 2];
		outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
	}
}


__global__ void edgeDetectionKernel(uint8_t * inPixels, int width, int height, 
	int filterWidth, 
	int * outPixels, int w, int h)
{

	extern __shared__ uint8_t s_inPixels[];

    int padding=filterWidth/2;
    int idx=(threadIdx.y * blockDim.x + threadIdx.x);
    //chia deu cac phan tu cho moi thread thuc hien copy
    while (idx < w * h)
    {
        int inR = blockIdx.y * blockDim.y - padding + idx / w;
        int inC = blockIdx.x * blockDim.x - padding + idx % w;
        inR = min(max(0, inR), height - 1);
        inC = min(max(0, inC), width - 1);
        s_inPixels[idx] = inPixels[inR * width + inC];
        idx+=(blockDim.x * blockDim.y);
    }


    __syncthreads();

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < height && c < width)
    { 
        int outPixel_X = 0;
        int outPixel_Y = 0;
        for (int filterR = 0; filterR < filterWidth; filterR++)
        {
            for (int filterC = 0; filterC < filterWidth; filterC++)
            {
                
                int inPixelsR = threadIdx.y + filterR;
                int inPixelsC = threadIdx.x + filterC;

                if (inPixelsR >= 0 && inPixelsR <= height - 1)
                    if (inPixelsC >= 0 && inPixelsC <= width - 1)
                    {
                        float filterVal_X = dc_filter_X[filterR*filterWidth + filterC];
                        float filterVal_Y = dc_filter_Y[filterR*filterWidth + filterC];

                        uint8_t inPixel = s_inPixels[inPixelsR*w + inPixelsC];

                        outPixel_X += filterVal_X * inPixel;
                        outPixel_Y += filterVal_Y * inPixel;
                    }   
            }
        }
        outPixels[r*width + c] = abs(outPixel_X) + abs(outPixel_Y);
    }

}

__global__ void compute_cost_v(int *energy, int width, int height) {

    // TODO
	int i = threadIdx.x + threadIdx.y*blockDim.x + gridDim.x*blockIdx.x;
	int min;
	if(i < width)
	{	
		for(int r = height-2; r >= 0; r--)
		{

			min = i + (r+1) * width;
			
			if(i-1 >= 0 && energy[i-1 + (r+1) * width] < energy[min])
				min = i-1 + (r+1) * width;

			if(i+1 <= width-1 && energy[i+1 + (r+1) * width] < energy[min])
				min = i+1 + (r+1) * width;

			energy[i + r * width] += energy[min];
			__syncthreads();
		}
	}

}


__global__ void findSeamKernel(int *energy, int *seam, int *next_pixels_v, int width, int height)
{
}

__global__ void removeSeamKernel(uint8_t * inPixels, int width, int height,
										int * next_pixels_v, uint8_t * outPixels)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

	if(j < width && i < height)
	{
		
		if (j + i * width < next_pixels_v[0])
		{
			outPixels[3*(j + i * width)] = inPixels[3*(j + i * width)];
			outPixels[3*(j + i * width)+1] = inPixels[3*(j + i * width)+1];
			outPixels[3*(j + i * width)+2] = inPixels[3*(j + i * width)+2];
		}
		else 
		{
			int temp = i;
			if (i == 0) 
			{
				temp = i + 1;
			}
			
			for(int idx = temp-1; idx < temp+3; idx++)
			{
				if(j + i * width + idx + 1 > width*height - 1) 
					break;

				if(j + i * width >= next_pixels_v[idx] - idx && j + i * width < next_pixels_v[idx+1] - idx - 1 && idx < height - 1)
				{
					outPixels[3*(j + i * width)] = inPixels[3*(j + i * width + idx + 1)];
					outPixels[3*(j + i * width)+1] = inPixels[3*(j + i * width + idx + 1)+1];
					outPixels[3*(j + i * width)+2] = inPixels[3*(j + i * width + idx + 1)+2];
				}

				if (j + i * width >= next_pixels_v[idx] - idx && idx == height-1)
				{
					outPixels[3*(j + i * width)] = inPixels[3*(j + i * width + (idx + 1))];
					outPixels[3*(j + i * width)+1] = inPixels[3*(j + i * width + (idx + 1))+1];
					outPixels[3*(j + i * width)+2] = inPixels[3*(j + i * width + (idx + 1))+2];
				}
			}
			
		}	
	}
		
}		



void seamCarvingKernel(uint8_t * inPixels, int width, int height, uint8_t * outPixels,
			float * filterX, float * filterY, int filterWidth, dim3 blockSize=dim3(1, 1),
        	int times = 1)
{
	// TODO
	uint8_t * d_inGrayscale, * inGrayscale, * d_inPixels, * d_outPixels;
	int * energy = (int*)malloc(width*height*sizeof(int));
	int * seam = (int*)malloc(height*sizeof(int));
	int * next_pixels_v = (int*)malloc(height*sizeof(int));
    int * cost_v = (int*)malloc(width*height*sizeof(int));
	int * d_energy, *d_next_pixels_v, *d_seam, *d_cost_v;
	float * d_filterX, * d_filterY;

	inGrayscale = (uint8_t*)malloc(width*height*sizeof(uint8_t));
	CHECK(cudaMalloc(&d_inPixels, width*height*3*sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_outPixels, width*height*3*sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_inGrayscale, width*height*sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_energy, width*height*sizeof(int)));
    CHECK(cudaMalloc(&d_cost_v, width*height*sizeof(int)));
	CHECK(cudaMalloc(&d_next_pixels_v, height*sizeof(int)));
	CHECK(cudaMalloc(&d_seam, height*sizeof(int)));
	CHECK(cudaMalloc(&d_filterX, filterWidth*filterWidth*sizeof(float)));	
	CHECK(cudaMalloc(&d_filterY, filterWidth*filterWidth*sizeof(float)));	

	dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
	dim3 gridSize_flatten((width-1)/(blockSize.x*blockSize.y) + 1);

	CHECK(cudaMemcpy(d_filterX, filterX, filterWidth*filterWidth*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_filterY, filterY, filterWidth*filterWidth*sizeof(float), cudaMemcpyHostToDevice));
	
	int w = blockSize.x + filterWidth - 1;
    int h = blockSize.y + filterWidth - 1;
	int sharedSize = w * h * sizeof(uint8_t);
		
	size_t nBytesFilter = filterWidth * filterWidth * sizeof(float);
	cudaMemcpyToSymbol(dc_filter_X, filterX, nBytesFilter);
    cudaMemcpyToSymbol(dc_filter_Y, filterY, nBytesFilter);
	CHECK(cudaMemcpy(d_inPixels, inPixels, width*height*3*sizeof(uint8_t), cudaMemcpyHostToDevice));

	for(int count = 0; count < times; count++)
	{
		convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_inGrayscale);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

        //Calculate energy of each pixel using Edge Detection
		
		edgeDetectionKernel<<<gridSize, blockSize, sharedSize>>>(d_inGrayscale, width, height, filterWidth, d_energy, w, h);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());


		compute_cost_v<<<gridSize_flatten, blockSize>>>(d_energy, width, height);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());


		CHECK(cudaMemcpy(cost_v, d_energy, width*height*sizeof(int), cudaMemcpyDeviceToHost));
        
		// Tìm phần tử có giá trị nhỏ nhất trên hàng cuối cùng của ma trận cost_v
		int min_col = 0;
		int min_val = cost_v[0 * width + 0];
		
		for (int j = 1; j < width; j++) {
			if (cost_v[0 * width + j] < min_val) {
				min_val = cost_v[0 * width + j];
				min_col = j;
			}
		}
		
		seam[0] = min_col;
		next_pixels_v[0] = seam[0];
		
		for (int i = 1; i < height ; i++) 
		{
			int j = seam[i-1];
			int min_j = j;
			if (j > 0 && cost_v[i * width + j-1] < cost_v[i * width + min_j]) {
				min_j = j-1;
			}
			if (j < width-1 && cost_v[i * width + j+1] < cost_v[i * width + min_j]) {
				min_j = j+1;
			}
			seam[i] = min_j;
			next_pixels_v[i] = i * width + seam[i];
		}
		

		CHECK(cudaMemcpy(d_next_pixels_v, next_pixels_v, height*sizeof(int), cudaMemcpyHostToDevice));
		removeSeamKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_next_pixels_v, d_outPixels);
		cudaDeviceSynchronize(); 
		CHECK(cudaGetLastError());
		

		width--;

		CHECK(cudaMemcpy(d_inPixels, d_outPixels, width*height*3*sizeof(uint8_t), cudaMemcpyDeviceToDevice));
	}
	CHECK(cudaMemcpy(outPixels, d_inPixels, width*height*3*sizeof(uint8_t), cudaMemcpyDeviceToHost));
	writePnm(outPixels, 3, width, height, "v3_device_out.pnm");
}
			
void seamCarvingHost(uint8_t * inPixels, int width, int height, uint8_t* outPixels, 
	float * filterX, float * filterY, int filterWidth, int times)
{
	
	int temp = filterWidth / 2;
	uint8_t * inGrayscale = (uint8_t*)malloc(width*height*sizeof(uint8_t));
	int * energy = (int*)malloc(width*height*sizeof(int));
	int * cost_v = (int*)malloc(width*height*sizeof(int));
	int * next_pixels_v = (int*)malloc(width*(height-1)*sizeof(int));
	

	for(int count = 0; count < times; count++)
	{
		//Convert RGB to Grayscale
		for (int r = 0; r < height; r++)
		{
			for (int c = 0; c < width; c++)
			{
				int i = r * width + c;
				uint8_t red = inPixels[3 * i];
				uint8_t green = inPixels[3 * i + 1];
				uint8_t blue = inPixels[3 * i + 2];
				inGrayscale[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
			}
		}		
		//Calculate importancy of each pixel using Edge Detection
		for (int resultR = 0; resultR < height; resultR++)
		{
			for (int resultC = 0; resultC < width; resultC++)
			{
				float importancy_X = 0.0f, importancy_Y = 0.0f;

				for (int filterR = 0; filterR < filterWidth; filterR++)
				{
					for (int filterC = 0; filterC < filterWidth; filterC++)
					{
						float filterValX = filterX[filterR*filterWidth + filterC];
						float filterValY = filterY[filterR*filterWidth + filterC];

						int inPixelsR = resultR - temp + filterR;
						int inPixelsC = resultC - temp + filterC;
						
						inPixelsR = min(max(0, inPixelsR), height - 1);
						inPixelsC = min(max(0, inPixelsC), width - 1);

						uint8_t inPixel = inGrayscale[inPixelsR*width + inPixelsC];

						importancy_X += filterValX * inPixel;
						importancy_Y += filterValY * inPixel;
					}
				}
				energy[resultR*width + resultC] = abs(importancy_X) + abs(importancy_Y);
			}
		}
		

		for (int j = 0; j < width; j++) 
		{
        	cost_v[(height - 1) * width + j] = energy[(height - 1) * width + j];
    	}

		// Duyệt các hàng tiếp theo của energy từ trên xuống dưới và từ trái sang phải
		for (int i = height - 2; i >= 0; i--) {
			for (int j = 0; j < width; j++) {
				// Tính giá trị cost_v[i][j]
				if (j == 0) {
					// Nếu j là cột đầu tiên, lấy giá trị bên phải của cost_v[i-1][j]
					cost_v[i * width + j] = energy[i * width + j] + fmin(cost_v[(i+1) * width + j], cost_v[(i+1) * width + j+1]);
				} else if (j == width-1) {
					// Nếu j là cột cuối cùng, lấy giá trị bên trái của cost_v[i-1][j]
					cost_v[i * width + j] = energy[i * width + j] + fmin(cost_v[(i+1) * width + j-1], cost_v[(i+1) * width + j]);
				} else {
					// Nếu j không nằm ở cột đầu tiên hoặc cuối cùng, lấy giá trị nhỏ nhất của hai phần tử bên trái và bên phải của nó trong hàng trên cùng của ma trận cost_v rồi cộng với energy[i][j]
					cost_v[i* width + j] = energy[i * width + j] + fmin(fmin(cost_v[(i+1) * width + j-1], cost_v[(i+1) * width + j]), cost_v[(i+1) * width + j+1]);
				}
			}
		}


		//Find min seam
		size_t seamSize = height*sizeof(int);
		if(width > height) seamSize = width*sizeof(int);
		int * seam = (int*)malloc(seamSize);


		// Tìm phần tử có giá trị nhỏ nhất trên hàng cuối cùng của ma trận cost_v
		int min_col = 0;
		int min_val = cost_v[0 * width + 0];
		
		for (int j = 1; j < width; j++) {
			if (cost_v[0 * width + j] < min_val) {
				min_val = cost_v[0 * width + j];
				min_col = j;
			}
		}
		
		seam[0] = min_col;
		next_pixels_v[0] = seam[0];
		
		for (int i = 1; i < height ; i++) 
		{
			int j = seam[i-1];
			int min_j = j;
			if (j > 0 && cost_v[i * width + j-1] < cost_v[i * width + min_j]) {
				min_j = j-1;
			}
			if (j < width-1 && cost_v[i * width + j+1] < cost_v[i * width + min_j]) {
				min_j = j+1;
			}
			seam[i] = min_j;
			next_pixels_v[i] = i * width + seam[i];
		}

		int idx = 0;
		//Remove min seam from the image vertically
		for(int i = 0; i < height; i++)
			{
				for(int j = 0; j < width; j++)
				{
					
					if (j + i * width >= next_pixels_v[idx] - idx && j + i * width <= next_pixels_v[height-1] - idx){
						idx++;
					}

					if(j + i * width + idx > width*height - 1) break;
						
					outPixels[3*(j + i * width)] = inPixels[3*(j + i * width + idx)];
					outPixels[3*(j + i * width)+1] = inPixels[3*(j + i * width + idx)+1];
					outPixels[3*(j + i * width)+2] = inPixels[3*(j + i * width + idx)+2];
					
					
				}
			}

		inPixels = outPixels;
		width--;
	}
	writePnm(outPixels, 3, width, height, "v3_host_out.pnm");
	
}

void seamCarving(uint8_t * inPixels, int width, int height, float * filterX, float * filterY, int filterWidth,
        uint8_t * outPixels, int times=1,
        bool useDevice=false, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer; 
	timer.Start();
	if (useDevice == false) // Use host
	{ 
		seamCarvingHost(inPixels, width, height, outPixels, filterX, filterY, filterWidth, times);
	}
	else // Use device
	{    
		seamCarvingKernel(inPixels, width, height, outPixels, filterX, filterY, filterWidth, blockSize, times);
	}
	timer.Stop();
	float time2 = timer.Elapsed();
	printf("%f ms\n", time2);
}

float computeError(uint8_t * a1, uint8_t * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i] - (int)a2[i]);
	}
		
	err /= n;
	return err;
}


char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n");

}

int main(int argc, char ** argv)
{
	if (argc == 4 && argc > 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	printDeviceInfo();

	// Read input image file
	int numChannels, width, height, times;
	uint8_t * inPixels, * inPixels_device, * inPixels_device_1;
	dim3 blockSize(32, 32);					

	readPnm(argv[1], numChannels, width, height, inPixels);
	
	char* type = argv[2];
	if (argc > 3)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	}
	if (argc == 6)
		times = atoi(argv[5]);

	if (numChannels != 3)
	{
		return EXIT_FAILURE; // Input image must be RGB
	}
	printf("\nImage size (width x height): %i x %i\n", width, height);

	// Set up a simple filter with blurring effect 
	int filterWidth = FILTER_WIDTH;

	float filterX[9] = {1.0, 0.0, -1.0,
                        2.0, 0.0, -2.0,
                        1.0, 0.0, -1.0};
    float filterY[9] = {1.0, 2.0, 1.0,
                        0.0, 0.0, 0.0,
                        -1.0, -2.0, -1.0};


	// Blur input image not using device
	uint8_t * correctOutPixels = (uint8_t *)malloc(width * height * numChannels * sizeof(uint8_t)); 
	uint8_t * outPixels = (uint8_t*)malloc(width * height * numChannels * sizeof(uint8_t));
	uint8_t * outPixels_1 = (uint8_t*)malloc(width * height * numChannels * sizeof(uint8_t));

	if(strcmp(type,"both") == 0)
	{
		// Seam carving by Host
		printf("Host time: \n");
		seamCarving(inPixels, width, height, filterX, filterY, filterWidth, correctOutPixels, times);

        //Seam carving by Device
		readPnm(argv[1], numChannels, width, height, inPixels_device);
		printf("Kernel time share memory: \n");
		seamCarving(inPixels_device, width, height, filterX, filterY, filterWidth, outPixels, times, true, blockSize);

		// Compute mean absolute error between host result and device result
		float err = computeError(outPixels, correctOutPixels, ((width-times) * height * numChannels));
		printf("Error between device result and host result: %f\n", err);

	}
	else if(strcmp(type,"kernel") == 0)
	{
		//Seam carving by Device
		printf("Kernel time: \n");
		seamCarving(inPixels, width, height, filterX, filterY, filterWidth, outPixels, times, true, blockSize);	
		

	}
	else if(strcmp(type,"host") == 0)
	{
		// Seam carving by Host
		printf("Host time: \n");
		seamCarving(inPixels, width, height, filterX, filterY, filterWidth, correctOutPixels, times);
		
	}

    // Free memories
	free(inPixels);
	free(correctOutPixels);
	free(outPixels);
}
