#include <stdio.h>
#include<ctime>
#include<iostream>
#include"cuda_runtime.h"

using namespace std;

int n=1024;//数据规模
float* data = NULL;//矩阵数据

//初始化矩阵数据
void init() {
	
	srand((int)time(0));
	for (int i = 0; i < n*n; i++) {
		data[i] = rand() % 100;
	}
}

//打印矩阵
void printMatrix() {
	for (int i = 0; i < n*n; i++) {
        if(i%n==0)
           cout<<endl;
        if(data[i]==-0)
            cout<<0<<"  ";
        else
            cout<<data[i]<<"  ";
	}
}


__global__ void division_kernel(float* data, int k, int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;//计算线程索引
    if(tid<N){
        int element = data[k*N+k];
        int temp = data[k*N+tid];
        //请同学们思考，如果分配的总线程数小于 N 应该怎么办？

        data[k*N+tid] = (float)temp/element;
    }
    
    return;
}

__global__ void eliminate_kernel(float* data, int k, int N){
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if(tx==0)
    data[k*N+k]=1.0;//对角线元素设为 1

    int row = k+1+blockIdx.x;//每个块负责一行

    while(row<N){
        int tid = threadIdx.x;
        while(k+1+tid < N){
            int col = k+1+tid;
            float temp_1 = data[(row*N) + col];
            float temp_2 = data[(row*N)+k];
            float temp_3 = data[k*N+col];
            data[(row*N) + col] = temp_1 - temp_2*temp_3;
            tid = tid + blockDim.x;
        }
        __syncthreads();//块内同步
        if (threadIdx.x == 0){
            data[row * N + k] = 0;
        }
        row += gridDim.x;
    }
    return;
}



int main(){
    cudaMallocManaged(&data,n*n*sizeof(float));//为data分配内存
    init();
   
    
    
    cudaError_t ret;
    
    cudaEvent_t start, stop;//计时器
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);//开始计时
    
    for(int k=0;k<n;k++){
        size_t threads_per_block=256;
       size_t number_of_blocks=(n+threads_per_block-1)/threads_per_block;
       
        division_kernel<<<number_of_blocks,threads_per_block>>>(data,k,n);//负责除法任务的核函数
        cudaDeviceSynchronize();//CPU 与 GPU 之间的同步函数
        ret = cudaGetLastError();
        if(ret!=cudaSuccess){
            printf("division_kernel failed, %s\n",cudaGetErrorString(ret));
        }
        
       
       
       eliminate_kernel<<<number_of_blocks,threads_per_block>>>(data,k,n);//负责消去任务的核函数
       cudaDeviceSynchronize();
       ret = cudaGetLastError();
        if(ret!=cudaSuccess){
            printf("eliminate_kernel failed, %s\n",cudaGetErrorString(ret));
        }  
        
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);//停止计时
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    
    
    cout<<endl;
    printf("GPU_LU:%f ms\n", elapsedTime);
    
    cudaFree ( data ) ; //释 放 data 内 存

    return 0;
}
