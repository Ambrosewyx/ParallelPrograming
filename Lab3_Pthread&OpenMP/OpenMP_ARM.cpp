#include<omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include<sys/time.h>
#include<arm_neon.h>
#include <semaphore.h>
using namespace std;


int n=10;//���ݹ�ģ
float** m_data ;//��������
const int thread_count=7;//�߳���

//��ʼ����������
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

//��ӡ����
void printMatrix(float**m_data) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << m_data[i][j] << " ";
		}
		cout << endl;
	}
}

//��ͨ��˹��ȥ�����㷨
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



//test��dynamicģʽ ���л��֣�(����黮��)
float** dynamic_row_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
	#pragma omp parallel if(parallel), num_threads(thread_count), private(i, j, k, temp)
	for ( k = 0; k < n; k++)
	{	
		//���в��֣�Ҳ���Կ��ǲ��л�
		#pragma omp single
		temp = m_data[k][k];
		for ( j = k + 1; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = 1.0;
		//���в��֣�ʹ���л���
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
		//�뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
	return m_data;
}


//test:dynamicģʽ + Neon ���л��֣�������黮�֣�
float** dynamic_neon_row_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	float32x4_t t1, t0;
	float32x4_t t2, t3, t4, vx;
	//����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
#pragma omp parallel if(parallel), num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//���в��֣�Ҳ���Կ��ǲ��л�
#pragma omp single
		temp = m_data[k][k];
		t0 =vdupq_n_f32(m_data[k][k]);
		for ( j = k + 1; j + 4 < n; j = j + 4)
		{
			t1 = vld1q_f32(m_data[k] + j);
			t1 = vdivq_f32(t1, t0);
			vst1q_f32(m_data[k] + j, t1);
		}
		for (int j = (n / 4) * 4; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = float(1.0);
		//���в��֣�ʹ���л���
#pragma omp for schedule(dynamic,n/thread_count)
		for ( i = k+1; i < n; i++)
		{
			temp = m_data[i][k];
			t2 = vdupq_n_f32(m_data[i][k]);
			for (int j = k + 1; j + 4 < n; j = j + 4)
			{
				t3 = vld1q_f32(m_data[k] + j);
				t4 = vld1q_f32(m_data[i] + j);
				vx = vmulq_f32(t2, t3);
				t4 = vsubq_f32(t4, vx);
				vst1q_f32(m_data[i] + j, t4);
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


//test: staticģʽ ���л��֣�������黮�֣�
float** static_row_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
#pragma omp parallel if(parallel), num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//���в��֣�Ҳ���Կ��ǲ��л�
#pragma omp single
		temp = m_data[k][k];
		for (j = k + 1; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = 1.0;
		//���в��֣�ʹ���л���
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
		//�뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
	return m_data;
}


//test��staticģʽ + Neon ���л��֣�������黮�֣�
float** static_neon_row_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	float32x4_t t1, t0;
	float32x4_t t2, t3, t4, vx;
	//����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
#pragma omp parallel if(parallel), num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//���в��֣�Ҳ���Կ��ǲ��л�
#pragma omp single
		temp = m_data[k][k];
		
		t0 = vdupq_n_f32(m_data[k][k]);
		for (j = k + 1; j + 4 < n; j = j + 4)
		{
			t1 = vld1q_f32(m_data[k] + j);
			t1 = vdivq_f32(t1, t0);
			vst1q_f32(m_data[k] + j, t1);
		}
		for (int j = (n / 4) * 4; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = float(1.0);
		//���в��֣�ʹ���л���
#pragma omp for schedule(static,n/thread_count)
		
		for (i = k + 1; i < n; i++)
		{
			temp = m_data[i][k];
			t2 = vdupq_n_f32(m_data[i][k]);
			for (int j = k + 1; j + 4 < n; j = j + 4)
			{
				t3 = vld1q_f32(m_data[k] + j);
				t4 = vld1q_f32(m_data[i] + j);
				vx = vmulq_f32(t2, t3);
				t4 = vsubq_f32(t4, vx);
				vst1q_f32(m_data[i] + j, t4);
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


//test: staticģʽ ���л��֣���ѭ�����֣�
float** static_row_loop_test(float**m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
#pragma omp parallel if(parallel), num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//���в��֣�Ҳ���Կ��ǲ��л�
#pragma omp single
		temp = m_data[k][k];
		for (j = k + 1; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = 1.0;
		//���в��֣�ʹ���л���
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
		//�뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
	return m_data;
}


//test: staticģʽ (�л���) ������黮�֣�
float** static_col_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
#pragma omp parallel if(parallel), num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//���в��֣�Ҳ���Կ��ǲ��л�
#pragma omp single
		temp = m_data[k][k];
		for (j = k + 1; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = 1.0;
		//���в��֣�ʹ���л���

		for (i = k + 1; i < n; i++)
		{
			temp = m_data[i][k];
#pragma omp for schedule(static,n/thread_count)
			
			for (j = k + 1; j < n; ++j)
			{
				m_data[i][j] = m_data[i][j] - temp * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}
		//�뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
	return m_data;
}


//test��guidedģʽ  ���л��֣� ������黮��--��С��Ϊn/thread_count��
float** guided_row_block_test(float** m_data)
{
	int k;
	int i;
	int j;
	float temp;
	//����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
#pragma omp parallel if(parallel), num_threads(thread_count), private(i, j, k, temp)
	for (k = 0; k < n; k++)
	{
		//���в��֣�Ҳ���Կ��ǲ��л�
#pragma omp single
		temp = m_data[k][k];
		for (j = k + 1; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / temp;
		}
		m_data[k][k] = 1.0;
		//���в��֣�ʹ���л���
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
		//�뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
	return m_data;
}



int main() {
	
	int m[9] = { 100,200,300,400,500,1000,2000,3000,4000 };
	for (int i = 0; i < 9; i++) {
		n = m[i];
		
		m_data = new float* [n];//��������

		//������ȷ��

		init();
		//printMatrix(m_data);
		//cout << endl;
		//printMatrix(normal_Gauss(m_data));						//��ͨ��˹
		//cout << endl;
		//printMatrix(dynamic_row_block_test(m_data));			//��̬  �л���  �黮��
		//printMatrix(dynamic_sse_row_block_test(m_data));		//��̬  SSE  �л���  �黮��
		//printMatrix(static_row_block_test(m_data));				//��̬  �л���  �黮��
		//printMatrix(static_sse_row_block_test(m_data));			//��̬  SSE  �л���  �黮��
		//printMatrix(static_row_loop_test(m_data));				//��̬  �л���  ѭ������
		//printMatrix(static_col_block_test(m_data));				//��̬  �л���  �黮��
		printMatrix(guided_row_block_test(m_data));				//guided �л���  �黮�֣���С��Ϊn/thread_count)--------���ز���


		//Linux��ȷ��ʱ
		struct timespec sts, ets;
		timespec_get(&sts, TIME_UTC);

		normal_Gauss(m_data);//�����㷨
		//dynamic_row_block_test(m_data);			//��̬  �л���  �黮��
		//dynamic_neon_row_block_test(m_data);		//��̬  Neon  �л���  �黮��
		//static_row_block_test(m_data);			//��̬  �л���  �黮��
		//static_neon_row_block_test(m_data);		//��̬  Neon  �л���  �黮��
		//static_row_loop_test(m_data);				//��̬  �л���  ѭ������
		//static_col_block_test(m_data);			//��̬  �л���  �黮��
		//guided_row_block_test(m_data);			//guided �л���  �黮�֣���С��Ϊn/thread_count)--------���ز���

		timespec_get(&ets, TIME_UTC);
		time_t dsec = ets.tv_sec - sts.tv_sec;
		long dnsec = ets.tv_nsec - sts.tv_nsec;
		if (dnsec < 0) {
			dsec--;
			dnsec += 1000000000ll;
		}
		printf(" % lld. % 09llds\n", dsec, dnsec);


		//�ͷŶ�̬�ڴ�
		for (int i = 0; i < n; i++) {
			delete[] m_data[i];
		}
		delete[] m_data;
	}

	return 0;
}