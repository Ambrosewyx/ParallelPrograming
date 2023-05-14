#include <pthread.h>
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
#include <semaphore.h>
using namespace std;

#pragma comment(lib, "pthreadVC2.lib")



int n=3000;//���ݹ�ģ
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
void printMatrix() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << m_data[i][j] << " ";
		}
		cout << endl;
	}
}

//��ͨ��˹��ȥ�����㷨
float** normal_Gauss() {
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

//��̬ר��
typedef struct {
	int k; //��ȥ���ִ�
	int t_id; // �߳� id
}threadParam_t;

//��̬ר��
typedef struct {
	int t_id;//�߳�id
}threadParam_t2;

//��̬���߳�
void* dynamic(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k; //��ȥ���ִ�
	int t_id = p->t_id; //�̱߳��
	int q = (n - k - 1) / thread_count;//�����������
	int w = k + t_id * q + 1; //��ȡ�Լ����׸���������
	for (int i = w; i < w + q; i++)
	{
		for (int j = k + 1; j < n; ++j)
		{
			m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
		}
		m_data[i][k] = 0;
	}
	pthread_exit(NULL);
	return NULL;
}

//test����̬���̣߳��л��֣�
void dynamic_test()
{
	for (int k = 0; k < n; k++)
	{	//���߳�����������
		for (int j = k + 1; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / m_data[k][k];
		}
		m_data[k][k] = 1.0;

		//���������̣߳�������ȥ����
		pthread_t* handles = new pthread_t[thread_count];// ������Ӧ�� Handle
		threadParam_t* param = new threadParam_t[thread_count];// ������Ӧ���߳����ݽṹ
		//��������
		for (int t_id = 0; t_id < thread_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//�����߳�
		for (int t_id = 0; t_id < thread_count; t_id++)
		{
			pthread_create(&handles[t_id], NULL, dynamic, (void*)&param[t_id]);
		}
		for (int i = k + (n - k - 1) / (thread_count+1) * thread_count+1 ; i < n; i++)
		{
			for (int j = k + 1; j < n; ++j)
			{
				m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
			}
			m_data[i][k] = 0;
		}
		//���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < thread_count; t_id++)
		{
			pthread_join(handles[t_id], NULL);
		}
	}
}

//��̬+SSE
void* dynamic_sse(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k; //��ȥ���ִ�
	int t_id = p->t_id; //�̱߳��
	int q = (n - k - 1) / thread_count;//�����������
	int w = k + t_id * q + 1; //��ȡ�Լ����׸���������
	__m128 t2, t3, t4, vx;
	for (int i = w; i < w + q; i++)
	{
		t2 = _mm_set_ps(m_data[i][k], m_data[i][k], m_data[i][k], m_data[i][k]);
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
			m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
		}
		m_data[i][k] = 0;
	}
	pthread_exit(NULL);
	return NULL;
}

//test: ��̬+SSE
void dynamic_sse_test() {
	for (int k = 0; k < n; k++)
	{	//���߳�����������
		__m128 t1, t0;
		t0 = _mm_set_ps(m_data[k][k], m_data[k][k], m_data[k][k], m_data[k][k]);
		for (int j = k + 1; j + 4 < n; j = j + 4)
		{
			t1 = _mm_loadu_ps(m_data[k] + j);
			t1 = _mm_div_ps(t1, t0);
			_mm_store_ps(m_data[k] + j, t1);
		}
		for (int j = (n / 4) * 4; j < n; j++)
		{
			m_data[k][j] = m_data[k][j] / m_data[k][k];
		}
		m_data[k][k] = float(1.0);
		//���������̣߳�������ȥ����
		pthread_t* handles = new pthread_t[thread_count];// ������Ӧ�� Handle
		threadParam_t* param = new threadParam_t[thread_count];// ������Ӧ���߳����ݽṹ
		//��������
		for (int t_id = 0; t_id < thread_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//�����߳�
		for (int t_id = 0; t_id < thread_count; t_id++)
		{
			pthread_create(&handles[t_id], NULL, dynamic_sse, (void*)&param[t_id]);
		}
		__m128 t2, t3, t4, vx;
		for (int i = k + (n - k - 1) / (thread_count+1) * thread_count + 1; i < n; i++)
		{
			t2 = _mm_set_ps(m_data[i][k], m_data[i][k], m_data[i][k], m_data[i][k]);
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
				m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
			}
			m_data[i][k] = 0;
		}
		//���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < thread_count; t_id++)
		{
			pthread_join(handles[t_id], NULL);
		}
	}
}


//��̬+�ź���ͬ��	����ѭ������	��ֱ����
//�ź�������
sem_t sem_main_loop;
sem_t sem_workerstart_loop[thread_count];//ÿ���߳����Լ�ר�����ź���
sem_t sem_workerend_loop[thread_count];
//�̺߳�������
void *static_loop(void* param) {
	threadParam_t2* p = (threadParam_t2*)param;
	int t_id = p->t_id;
	for (int k = 0; k < n; ++k) {
		sem_wait(&sem_workerstart_loop[t_id]);//�������ȴ�������ɳ��������������Լ�ר�����ź�����

		//ѭ�����񻮷�
		for (int i = k + 1 + t_id; i < n; i += thread_count) {
			//��ȥ
			for (int j = k + 1; j < n; j++) {
				m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}
		sem_post(&sem_main_loop);//�������߳�
		sem_wait(&sem_workerend_loop[t_id]);//�������ȴ����̻߳��ѽ�����һ��
	}
	pthread_exit(NULL);
	return NULL;
}

//test: ��̬+�ź���ͬ��	����ѭ������	��ֱ����
void static_loop_test() {
	//��ʼ���ź���
	sem_init(&sem_main_loop, 0, 0);
	for(int i = 0; i < thread_count; ++i) {
		sem_init(&sem_workerstart_loop[i], 0, 0);
		sem_init(&sem_workerend_loop[i], 0, 0);
	}
	//�����߳�
	pthread_t handles[thread_count];
	threadParam_t2 param[thread_count];
	for (int t_id = 0; t_id < thread_count; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, static_loop, (void*)&param[t_id]);
	}

	for (int k = 0; k < n; k++) {
		//���߳�����������
		for (int j = k + 1; j < n; j++) {
			m_data[k][j] = m_data[k][j] / m_data[k][k];
		}
		m_data[k][k] = 1.0;

		//��ʼ���ѹ����߳�
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_post(&sem_workerstart_loop[t_id]);
		}

		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_wait(&sem_main_loop);
		}

		//���߳��ٴλ��ѹ����߳̽�����һ�ֵ���ȥ����
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_post(&sem_workerend_loop[t_id]);
		}
	}
	for (int t_id = 0; t_id < thread_count; t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	//���������ź���
	sem_destroy(&sem_main_loop);
	sem_destroy(&sem_workerstart_loop[thread_count]);
	sem_destroy(&sem_workerend_loop[thread_count]);
}

//��̬+�ź���ͬ��	���񰴿黮��	��ֱ����
//�ź�������
sem_t sem_main_block;
sem_t sem_workerstart_block[thread_count];//ÿ���߳����Լ�ר�����ź���
sem_t sem_workerend_block[thread_count];
//�̺߳�������
void* static_block(void* param) {
	threadParam_t2* p = (threadParam_t2*)param;
	int t_id = p->t_id;
	for (int k = 0; k < n; ++k) {
		sem_wait(&sem_workerstart_block[t_id]);//�������ȴ�������ɳ��������������Լ�ר�����ź�����

		//���񰴿黮��
		int q = (n - k - 1) / (thread_count+1);
		int w = k + t_id * q + 1; //��ȡ�Լ��ļ�������
		for (int i = w; i < w + q; i++)
		{
			for (int j = k + 1; j < n; ++j)
			{
				m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
			}
			m_data[i][k] = 0;
		}
		sem_post(&sem_main_block);//�������߳�
		sem_wait(&sem_workerend_block[t_id]);//�������ȴ����̻߳��ѽ�����һ��
	}
	pthread_exit(NULL);
	return NULL;
}

//test: ��̬+�ź���ͬ��	���񰴿黮��	��ֱ����
void static_block_test() {
	//��ʼ���ź���
	sem_init(&sem_main_block, 0, 0);
	for(int i = 0; i < thread_count; ++i) {
		sem_init(&sem_workerstart_block[i], 0, 0);
		sem_init(&sem_workerend_block[i], 0, 0);
	}
	//�����߳�
	pthread_t handles[thread_count];
	threadParam_t2 param[thread_count];
	for (int t_id = 0; t_id < thread_count; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, static_block, (void*)&param[t_id]);
	}

	for (int k = 0; k < n; k++) {
		//���߳�����������
		for (int j = k + 1; j < n; j++) {
			m_data[k][j] = m_data[k][j] / m_data[k][k];
		}
		m_data[k][k] = 1.0;

		//��ʼ���ѹ����߳�
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_post(&sem_workerstart_block[t_id]);
		}
			for (int i = k + (n - k - 1) / (thread_count+1) * thread_count + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; ++j)
			{
				m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}

		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_wait(&sem_main_block);
		}

		//���߳��ٴλ��ѹ����߳̽�����һ�ֵ���ȥ����
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_post(&sem_workerend_block[t_id]);
		}
	}
	for (int t_id = 0; t_id < thread_count; t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	//���������ź���
	sem_destroy(&sem_main_block);
	sem_destroy(&sem_workerstart_block[thread_count]);
	sem_destroy(&sem_workerend_block[thread_count]);
}


//��̬+�ź���ͬ��	����ѭ������	ˮƽ����
//�ź�������
sem_t sem_main_loop2;
sem_t sem_workerstart_loop2[thread_count];//ÿ���߳����Լ�ר�����ź���
sem_t sem_workerend_loop2[thread_count];
//�̺߳�������
void* static_loop2(void* param) {
	threadParam_t2* p = (threadParam_t2*)param;
	int t_id = p->t_id;
	for (int k = 0; k < n; ++k) {
		sem_wait(&sem_workerstart_loop2[t_id]);//�������ȴ�������ɳ��������������Լ�ר�����ź�����

		//ѭ�����񻮷�	ˮƽ����
		for (int i = k + 1; i < n; i ++) {
			//��ȥ
			for (int j = k + 1+t_id; j < n; j+=thread_count) {
				m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}
		sem_post(&sem_main_loop2);//�������߳�
		sem_wait(&sem_workerend_loop2[t_id]);//�������ȴ����̻߳��ѽ�����һ��
	}
	pthread_exit(NULL);
	return NULL;
}

//test: ��̬+�ź���ͬ��	����ѭ������	ˮƽ����
void static_loop_test2() {
	//��ʼ���ź���
	sem_init(&sem_main_loop2, 0, 0);
	for (int i = 0; i < thread_count; ++i) {
		sem_init(&sem_workerstart_loop2[i], 0, 0);
		sem_init(&sem_workerend_loop2[i], 0, 0);
	}
	//�����߳�
	pthread_t handles[thread_count];
	threadParam_t2 param[thread_count];
	for (int t_id = 0; t_id < thread_count; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, static_loop2, (void*)&param[t_id]);
	}

	for (int k = 0; k < n; k++) {
		//���߳�����������
		for (int j = k + 1; j < n; j++) {
			m_data[k][j] = m_data[k][j] / m_data[k][k];
		}
		m_data[k][k] = 1.0;

		//��ʼ���ѹ����߳�
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_post(&sem_workerstart_loop2[t_id]);
		}

		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_wait(&sem_main_loop2);
		}

		//���߳��ٴλ��ѹ����߳̽�����һ�ֵ���ȥ����
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_post(&sem_workerend_loop2[t_id]);
		}
	}
	for (int t_id = 0; t_id < thread_count; t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	//���������ź���
	sem_destroy(&sem_main_loop2);
	sem_destroy(&sem_workerstart_loop2[thread_count]);
	sem_destroy(&sem_workerend_loop2[thread_count]);
}


//��̬+�ź���ͬ��+SSE	���񰴿黮��	��ֱ����
//�ź�������
sem_t sem_main_block2;
sem_t sem_workerstart_block2[thread_count];//ÿ���߳����Լ�ר�����ź���
sem_t sem_workerend_block2[thread_count];
//�̺߳�������
void* static_block2(void* param) {
	threadParam_t2* p = (threadParam_t2*)param;
	int t_id = p->t_id;
	for (int k = 0; k < n; ++k) {
		sem_wait(&sem_workerstart_block2[t_id]);//�������ȴ�������ɳ��������������Լ�ר�����ź�����

		//���񰴿黮��
		int q = (n - k - 1) / (thread_count + 1);
		int w = k + t_id * q + 1; //��ȡ�Լ��ļ�������
		__m128 t2, t3, t4, vx;
		for (int i = w; i < w + q; i++)
		{
			t2 = _mm_set_ps(m_data[i][k], m_data[i][k], m_data[i][k], m_data[i][k]);
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
				m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
			}
			m_data[i][k] = 0;
		}
		sem_post(&sem_main_block2);//�������߳�
		sem_wait(&sem_workerend_block2[t_id]);//�������ȴ����̻߳��ѽ�����һ��
	}
	pthread_exit(NULL);
	return NULL;
}

//test: ��̬+�ź���ͬ��+SSE 	���񰴿黮��	��ֱ����
void static_block_sse_test() {
	//��ʼ���ź���
	sem_init(&sem_main_block2, 0, 0);
	for (int i = 0; i < thread_count; ++i) {
		sem_init(&sem_workerstart_block2[i], 0, 0);
		sem_init(&sem_workerend_block2[i], 0, 0);
	}
	//�����߳�
	pthread_t handles[thread_count];
	threadParam_t2 param[thread_count];
	for (int t_id = 0; t_id < thread_count; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, static_block2, (void*)&param[t_id]);
	}

	for (int k = 0; k < n; k++) {
		//���߳�����������
		for (int j = k + 1; j < n; j++) {
			m_data[k][j] = m_data[k][j] / m_data[k][k];
		}
		m_data[k][k] = 1.0;

		//��ʼ���ѹ����߳�
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_post(&sem_workerstart_block2[t_id]);
		}
		for (int i = k + (n - k - 1) / (thread_count + 1) * thread_count + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; ++j)
			{
				m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}

		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_wait(&sem_main_block2);
		}

		//���߳��ٴλ��ѹ����߳̽�����һ�ֵ���ȥ����
		for (int t_id = 0; t_id < thread_count; t_id++) {
			sem_post(&sem_workerend_block2[t_id]);
		}
	}
	for (int t_id = 0; t_id < thread_count; t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	//���������ź���
	sem_destroy(&sem_main_block2);
	sem_destroy(&sem_workerstart_block2[thread_count]);
	sem_destroy(&sem_workerend_block2[thread_count]);
}

//��̬+�ź���ͬ��+����ѭ��ȫ�������̺߳���	ѭ������	��ֱ����
//�ź�������
sem_t sem_leader;
sem_t sem_Division[thread_count - 1];
sem_t sem_Elimiination[thread_count - 1];
//�̺߳�������
void* static_loop3(void* param) {
	threadParam_t2* p = (threadParam_t2*)param;
	int t_id = p->t_id;
	for (int k = 0; k < n; k++) {
		//t_idΪ0���̳߳��������������߳��ȵȴ�
		if (t_id == 0) {
			for (int j = k + 1; j < n; j++) {
				m_data[k][j] = m_data[k][j] / m_data[k][k];
			}
			m_data[k][k] = 1.0;
		}
		else
			sem_wait(&sem_Division[t_id - 1]);//�������ȴ���ɳ�������
		//t_id Ϊ0���̻߳������������̣߳�������ȥ����
		if (t_id == 0) {
			for (int i = 0; i < thread_count - 1; i++) {
				sem_post(&sem_Division[i]);
			}
		}
		//ѭ�����񻮷�
		for (int i = k + 1 + t_id; i < n; i += thread_count) {
			//��ȥ
			for (int j = k + 1; j < n; j++) {
				m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}

		if (t_id == 0) {
			for (int i = 0; i < thread_count - 1; ++i) {
				sem_wait(&sem_leader);//�ȴ�����worker�����ȥ
			}

			for (int i = 0; i < thread_count - 1; i++) {
				sem_post(&sem_Elimiination[i]);//֪ͨ����worker������һ��
			}
		}
		else {
			sem_post(&sem_leader);
			sem_wait(&sem_Elimiination[t_id - 1]);
		}
	}
	pthread_exit(NULL);
	return NULL;
}

//test: ��̬+�ź���ͬ��+����ѭ��ȫ�������̺߳���  ѭ������  ��ֱ����
void static_loop_test3() {
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < thread_count - 1; i++) {
		sem_init(&sem_Division[i], 0, 0);
		sem_init(&sem_Elimiination[i], 0, 0);
	}
	//�����߳�
	pthread_t handles[thread_count];
	threadParam_t2 param[thread_count];
	for (int t_id = 0; t_id < thread_count; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, static_loop3, (void*)&param[t_id]);
	}
	
	for (int t_id = 0; t_id < thread_count; t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	//�����ź���
	sem_destroy(&sem_leader);
	sem_destroy(&sem_Division[thread_count-1]);
	sem_destroy(&sem_Elimiination[thread_count-1]);
}

//��̬+barrierͬ��	ѭ������  ��ֱ����
//barrier����
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
//�̺߳�������
void* static_barrier_loop(void* param) {
	threadParam_t2* p = (threadParam_t2*)param;
	int t_id = p->t_id;
	for (int k = 0; k < n; k++) {
		//t_idΪ0���̳߳��������������߳��ȵȴ�
		if (t_id == 0) {
			for (int j = k + 1; j < n; j++) {
				m_data[k][j] = m_data[k][j] / m_data[k][k];
			}
			m_data[k][k] = 1.0;
		}

		//��һ��ͬ����
		pthread_barrier_wait(&barrier_Divsion);
		//ѭ�����񻮷�
		for (int i = k + 1 + t_id; i < n; i += thread_count) {
			//��ȥ
			for (int j = k + 1; j < n; j++) {
				m_data[i][j] = m_data[i][j] - m_data[i][k] * m_data[k][j];
			}
			m_data[i][k] = 0.0;
		}
		
		//�ڶ���ͬ����
		pthread_barrier_wait(&barrier_Elimination);
	}
	pthread_exit(NULL);
	return NULL;
}

//test: ��̬+barrierͬ��	ѭ������  ��ֱ����
void static_barrier_loop_test() {
	pthread_barrier_init(&barrier_Divsion, NULL, thread_count);
	pthread_barrier_init(&barrier_Elimination, NULL, thread_count);
	//�����߳�
	pthread_t handles[thread_count];
	threadParam_t2 param[thread_count];
	for (int t_id = 0; t_id < thread_count; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, static_barrier_loop, (void*)&param[t_id]);
	}

	for (int t_id = 0; t_id < thread_count; t_id++) {
		pthread_join(handles[t_id], NULL);
	}

	//����barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);
}

int main() {
	
	int m[9] = { 100,200,300,400,500,1000,2000,3000,4000 };
	for (int i = 0; i < 9; i++) {
		n = m[i];
		
		m_data = new float* [n];//��������
		init();
		//printMatrix();
		long long head, tail, freq;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);

		//normal_Gauss();//�����㷨
		dynamic_test();//��̬�̣߳���ֱ���֣�
		//dynamic_sse_test();//��̬+SSE

		//static_block_test();//��̬+�ź���ͬ��	����黮��	��ֱ����
		//static_block_sse_test();//��̬+�ź���ͬ��+SSE 	���񰴿黮��	��ֱ����

		//static_loop_test2();//��̬+�ź���ͬ��	����ѭ������	ˮƽ����
		//static_loop_test();//��̬+�ź���ͬ��	����ѭ������	��ֱ����
		//static_loop_test3();//��̬+�ź���ͬ��+����ѭ��ȫ�������̺߳���  ѭ������  ��ֱ����
		//static_barrier_loop_test();//��̬+barrierͬ��  ѭ������  ��ֱ����

		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << (tail - head) * 1.0 / freq << endl;

		//�ͷŶ�̬�ڴ�
		for (int i = 0; i < n; i++) {
			delete[] m_data[i];
		}
		delete[] m_data;
	}

	return 0;
}