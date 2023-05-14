#include<omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <iostream>
#include <ctime>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
using namespace std;


int n=2000;//数据规模
float** m_data ;//矩阵数据
const int thread_count=7;//线程数

//初始化矩阵数据
void init() {
	for (int i = 0; i < n; i++) {
		m_data[i] = new float[n];
	}
	srand((int)time(0));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			m_data[i][j] = rand() % 100;
		}
	}
}

//打印矩阵
void printMatrix(float**m_data) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << m_data[i][j] << " ";
		}
		cout << endl;
	}
}

//普通高斯消去串行算法
float** normal_Gauss(float** m_data) {
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			m_data[k][j] = m_data[k][j] / m_data[k][k];
		}
		m_data[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++) {
				m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
			}
			m_data[i][k] = 0;
		}
	}
	return m_data;
}



//test：dynamic模式 （行划分）(任务块划分)
float** dynamic_row_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	#pragma omp parallel , num_threads(thread_count), private(i, j, k, temp)
	for ( k = 0; k < n; k++)
	{	
		//串行部分，也可以考虑并行化
		#pragma omp single
		temp = m_data[k][k];
		for ( j = k + 1; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = 1.0;
		//并行部分，使用行划分
		#pragma omp for schedule(dynamic,n/thread_count)
		for ( i = k +1 ; i < n; i++)
		{
			temp = m_data[i][k];
			for ( j = k + 1; j < n; ++j)
			{
				m_data[i][j] = m_data[i][j] -temp * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}
		//离开for循环时，各个线程默认同步，进入下一行的处理
	}
	return m_data;
}


//test:dynamic模式 + SSE （行划分）（任务块划分）
float** dynamic_sse_row_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
#pragma omp parallel , num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//串行部分，也可以考虑并行化
#pragma omp single
		temp = m_data[k][k];
		__m128 t1, t0;
		t0 = _mm_set_ps(temp,temp,temp,temp);
		for ( j = k + 1; j + 4 < n; j = j + 4)
		{
			t1 = _mm_loadu_ps(m_data[k] + j);
			t1 = _mm_div_ps(t1, t0);
			_mm_store_ps(m_data[k] + j, t1);
		}
		for (int j = (n / 4) * 4; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = float(1.0);
		//并行部分，使用行划分
#pragma omp for schedule(dynamic,n/thread_count)
		__m128 t2, t3, t4, vx;
		for ( i = k+1; i < n; i++)
		{
			temp = m_data[i][k];
			t2 = _mm_set_ps(temp,temp,temp,temp);
			for (int j = k + 1; j + 4 < n; j = j + 4)
			{
				t3 = _mm_loadu_ps(m_data[k] + j);
				t4 = _mm_loadu_ps(m_data[i] + j);
				vx = _mm_mul_ps(t2, t3);
				t4 = _mm_sub_ps(t4, vx);
				_mm_store_ps(m_data[i] + j, t4);
			}
			for (int j = (n / 4) * 4; j < n; j++)
			{
				m_data[i][j] = m_data[i][j] - temp * m_data[k][j];
			}
			m_data[i][k] = 0;
		}
	}
	return m_data;
}


//test: static模式 （行划分）（任务块划分）
float** static_row_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
#pragma omp parallel , num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//串行部分，也可以考虑并行化
#pragma omp single
		temp = m_data[k][k];
		for (j = k + 1; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = 1.0;
		//并行部分，使用行划分
#pragma omp for schedule(static,n/thread_count)
		for (i = k + 1; i < n; i++)
		{
			temp = m_data[i][k];
			for (j = k + 1; j < n; ++j)
			{
				m_data[i][j] = m_data[i][j] - temp * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}
		//离开for循环时，各个线程默认同步，进入下一行的处理
	}
	return m_data;
}


//test：static模式 + SSE （行划分）（任务块划分）
float** static_sse_row_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
#pragma omp parallel , num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//串行部分，也可以考虑并行化
#pragma omp single
		temp = m_data[k][k];
		__m128 t1, t0;
		t0 = _mm_set_ps(temp, temp, temp, temp);
		for (j = k + 1; j + 4 < n; j = j + 4)
		{
			t1 = _mm_loadu_ps(m_data[k] + j);
			t1 = _mm_div_ps(t1, t0);
			_mm_store_ps(m_data[k] + j, t1);
		}
		for (int j = (n / 4) * 4; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = float(1.0);
		//并行部分，使用行划分
#pragma omp for schedule(static,n/thread_count)
		__m128 t2, t3, t4, vx;
		for (i = k + 1; i < n; i++)
		{
			temp = m_data[i][k];
			t2 = _mm_set_ps(temp, temp, temp, temp);
			for (int j = k + 1; j + 4 < n; j = j + 4)
			{
				t3 = _mm_loadu_ps(m_data[k] + j);
				t4 = _mm_loadu_ps(m_data[i] + j);
				vx = _mm_mul_ps(t2, t3);
				t4 = _mm_sub_ps(t4, vx);
				_mm_store_ps(m_data[i] + j, t4);
			}
			for (int j = (n / 4) * 4; j < n; j++)
			{
				m_data[i][j] = m_data[i][j] - temp * m_data[k][j];
			}
			m_data[i][k] = 0;
		}
	}
	return m_data;
}


//test: static模式 （行划分）（循环划分）
float** static_row_loop_test(float**m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
#pragma omp parallel , num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//串行部分，也可以考虑并行化
#pragma omp single
		temp = m_data[k][k];
		for (j = k + 1; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = 1.0;
		//并行部分，使用行划分
#pragma omp for schedule(static,1)
		for (i = k + 1; i < n; i++)
		{
			temp = m_data[i][k];
			for (j = k + 1; j < n; ++j)
			{
				m_data[i][j] = m_data[i][j] - temp * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}
		//离开for循环时，各个线程默认同步，进入下一行的处理
	}
	return m_data;
}


//test: static模式 (列划分) （任务块划分）
float** static_col_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
#pragma omp parallel , num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//串行部分，也可以考虑并行化
#pragma omp single
		temp = m_data[k][k];
		for (j = k + 1; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = 1.0;
		//并行部分，使用列划分

		for (i = k + 1; i < n; i++)
		{
#pragma omp for schedule(static,n/thread_count)
			temp = m_data[i][k];
			for (j = k + 1; j < n; ++j)
			{
				m_data[i][j] = m_data[i][j] - temp * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}
		//离开for循环时，各个线程默认同步，进入下一行的处理
	}
	return m_data;
}


//test：guided模式  （行划分） （任务块划分--最小块为n/thread_count）
float** guided_row_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
#pragma omp parallel , num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//串行部分，也可以考虑并行化
#pragma omp single
		temp = m_data[k][k];
		for (j = k + 1; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = 1.0;
		//并行部分，使用行划分
#pragma omp for schedule(guided,n/thread_count)
		for (i = k + 1; i < n; i++)
		{
			temp = m_data[i][k];
			for (j = k + 1; j < n; ++j)
			{
				m_data[i][j] = m_data[i][j] - temp * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}
		//离开for循环时，各个线程默认同步，进入下一行的处理
	}
	return m_data;
}



int main() {
	
	int m[9] = { 100,200,300,400,500,1000,2000,3000,4000 };
	for (int i = 0; i < 9; i++) {
		n = m[i];
		
		m_data = new float* [n];//矩阵数据

		//检验正确性

		init();
		//printMatrix(m_data);
		//cout << endl;
		//printMatrix(normal_Gauss(m_data));						//普通高斯
		//cout << endl;
		//printMatrix(dynamic_row_block_test(m_data));			//动态  行划分  块划分
		//printMatrix(dynamic_sse_row_block_test(m_data));		//动态  SSE  行划分  块划分
		//printMatrix(static_row_block_test(m_data));				//静态  行划分  块划分
		//printMatrix(static_sse_row_block_test(m_data));			//静态  SSE  行划分  块划分
		//printMatrix(static_row_loop_test(m_data));				//静态  行划分  循环划分
		//printMatrix(static_col_block_test(m_data));				//静态  列划分  块划分
		//printMatrix(guided_row_block_test(m_data));				//guided 行划分  块划分（最小块为n/thread_count)--------负载不均


		//测试运行时间

		long long head, tail, freq;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);

		//normal_Gauss(m_data);//串行算法
		//dynamic_row_block_test(m_data);			//动态  行划分  块划分
		//dynamic_sse_row_block_test(m_data);		//动态  SSE  行划分  块划分
		//static_row_block_test(m_data);			//静态  行划分  块划分
		//static_sse_row_block_test(m_data);		//静态  SSE  行划分  块划分
		//static_row_loop_test(m_data);				//静态  行划分  循环划分
		//static_col_block_test(m_data);			//静态  列划分  块划分
		guided_row_block_test(m_data);			//guided 行划分  块划分（最小块为n/thread_count)--------负载不均

		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << (tail - head) * 1.0 / freq << endl;

		//释放动态内存
		for (int i = 0; i < n; i++) {
			delete[] m_data[i];
		}
		delete[] m_data;
	}

	return 0;
}