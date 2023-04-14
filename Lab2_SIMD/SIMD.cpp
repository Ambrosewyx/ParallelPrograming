#include<iostream>
#include<ctime>
#include<stdio.h>
#include<windows.h>
#include<time.h>
//#include<arm_neon.h>//Neon
#include<xmmintrin.h>//SSE
#include<immintrin.h>//AVX��AVX2
using namespace std;

//��ʼ����������
void init(float**& data,int n) {
	srand((int)time(0));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			data[i][j] = rand()%100;
		}
	}
}

//��ӡ����
void printMatrix(float** data, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << data[i][j] << " ";
		}
		cout << endl;
	}
}


//��ͨ��˹��ȥ�����㷨
float** normal_gauss(float** data, int n) {
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			data[k][j] = data[k][j] / data[k][k];
		}
		data[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++) {
				data[i][j] = data[i][j] - data[i][k] * data[k][j];
			}
			data[i][k] = 0;
		}
	}
	return data;
}

//��ͨ��˹��ȥSSE����
float** sse_gauss(float** data, int n) {
	__m128 t1, t2, t3, t4;

	for (int k = 0; k < n; k++)
	{
		//��������
		float tmp[4] = { data[k][k], data[k][k], data[k][k], data[k][k] };
		t1 = _mm_loadu_ps(tmp);
		for (int j = n - 4; j >= k; j -= 4) //�Ӻ���ǰÿ��ȡ�ĸ�
		{
			t2 = _mm_loadu_ps(data[k] + j);
			t3 = _mm_div_ps(t2, t1);//����
			_mm_storeu_ps(data[k] + j, t3);
		}

		if (k % 4 != (n % 4)) //�����ܱ�4������Ԫ��
		{
			for (int j = k; j % 4 != (n % 4); j++)
			{
				data[k][j] = data[k][j] / tmp[0];
			}
		}

		for (int j = (n % 4) - 1; j >= 0; j--)
		{
			data[k][j] = data[k][j] / tmp[0];
		}


		//��������
		for (int i = k + 1; i < n; i++)
		{
			float tmp[4] = { data[i][k], data[i][k], data[i][k], data[i][k] };
			t1 = _mm_loadu_ps(tmp);
			for (int j = n - 4; j > k; j -= 4)
			{
				t2 = _mm_loadu_ps(data[i] + j);
				t3 = _mm_loadu_ps(data[k] + j);
				t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3)); //����
				_mm_storeu_ps(data[i] + j, t4);
			}

			for (int j = k + 1; j % 4 != (n % 4); j++)
			{
				data[i][j] = data[i][j] - data[i][k] * data[k][j];
			}

			data[i][k] = 0;
		}
	}
	return data;
}

//��ͨ��˹��ȥNeon����
//float** neon_gauss(float** data, int n) {
//	float32x4_t t1, t2, t3, t4;
//
//	for (int k = 0; k < n; k++)
//	{
//		//��������
//		float tmp[4] = { data[k][k], data[k][k], data[k][k], data[k][k] };
//		t1 = vld1q_f32(tmp);
//		for (int j = n - 4; j >= k; j -= 4) //�Ӻ���ǰÿ��ȡ�ĸ�
//		{
//			t2 =  vld1q_f32(data[k] + j);
//			t3 = vdivq_f32(t2, t1);//����
//			vst1q_f32(data[k] + j, t3);
//		}
//
//		if (k % 4 != (n % 4)) //�����ܱ�4������Ԫ��
//		{
//			for (int j = k; j % 4 != (n % 4); j++)
//			{
//				data[k][j] = data[k][j] / tmp[0];
//			}
//		}
//
//		for (int j = (n % 4) - 1; j >= 0; j--)
//		{
//			data[k][j] = data[k][j] / tmp[0];
//		}
//
//
//		//��������
//		for (int i = k + 1; i < n; i++)
//		{
//			float tmp[4] = { data[i][k], data[i][k], data[i][k], data[i][k] };
//			t1 = vld1q_f32(tmp);
//			for (int j = n - 4; j > k; j -= 4)
//			{
//				t2 = vld1q_f32(data[i] + j);
//				t3 = vld1q_f32(data[k] + j);
//				t4 = vsubq_f32(t2, vmulq_f32(t1, t3)); //����
//				vst1q_f32(data[i] + j, t4);
//			}
//
//			for (int j = k + 1; j % 4 != (n % 4); j++)
//			{
//				data[i][j] = data[i][j] - data[i][k] * data[k][j];
//			}
//
//			data[i][k] = 0;
//		}
//	}
//	return data;
//}

//��ͨ��˹��ȥAVX����
float** avx_gauss(float** data, int n) {
	__m256 t1, t2, t3, t4;

	for (int k = 0; k < n; k++)
	{
		//��������
		float tmp[8] = { data[k][k], data[k][k], data[k][k], data[k][k],data[k][k], data[k][k], data[k][k], data[k][k] };
		t1 = _mm256_loadu_ps(tmp);
		for (int j = n - 8; j >= k; j -= 8) //�Ӻ���ǰÿ��ȡ�˸�
		{
			t2 = _mm256_loadu_ps(data[k] + j);
			t3 = _mm256_div_ps(t2, t1);//����
			_mm256_storeu_ps(data[k] + j, t3);
		}

		if (k % 8 != (n % 8)) //�����ܱ�4������Ԫ��
		{
			for (int j = k; j % 8 != (n % 8); j++)
			{
				data[k][j] = data[k][j] / tmp[0];
			}
		}

		for (int j = (n % 8) - 1; j >= 0; j--)
		{
			data[k][j] = data[k][j] / tmp[0];
		}


		//��������
		for (int i = k + 1; i < n; i++)
		{
			float tmp[8] = { data[i][k], data[i][k], data[i][k], data[i][k], data[i][k], data[i][k], data[i][k], data[i][k] };
			t1 = _mm256_loadu_ps(tmp);
			for (int j = n - 8; j > k; j -= 8)
			{
				t2 = _mm256_loadu_ps(data[i] + j);
				t3 = _mm256_loadu_ps(data[k] + j);
				t4 = _mm256_sub_ps(t2, _mm256_mul_ps(t1, t3)); //����
				_mm256_storeu_ps(data[i] + j, t4);
			}

			for (int j = k + 1; j % 8 != (n % 8); j++)
			{
				data[i][j] = data[i][j] - data[i][k] * data[k][j];
			}

			data[i][k] = 0;
		}
	}
	return data;
}

int main() {
	
	while (true) {
		int n;//�����ģ
		cin >> n;
		float** data = new float* [n];
		for (int i = 0; i < n; ++i) {
			data[i] = new float[n];
		}
		init(data, n);

		//���Խ���Ƿ���ȷ
		/*printMatrix(data, n);
		cout << endl;
		printMatrix(normal_gauss(data, n),n);
		cout << endl;
		printMatrix(sse_gauss(data, n), n);
		cout << endl;
		printMatrix(neon_gauss(data, n), n);
		cout << endl;
		printMatrix(avx_gauss(data, n), n);
		
		*/

		//Linux��ȷ��ʱ
		//struct timespec sts, ets;
		//timespec_get(&sts, TIME_UTC);
		//normal_gauss(data, n);//�����㷨
		////sse_gauss(data, n);//sse�㷨
		////neon_gauss(data, n);//neon�㷨
		//avx_gauss(data, n);//AVX�㷨
		//timespec_get(&ets, TIME_UTC);
		//time_t dsec = ets.tv_sec - sts.tv_sec;
		//long dnsec = ets.tv_nsec - sts.tv_nsec;
		//if (dnsec < 0) {
		//	dsec--;
		//	dnsec += 1000000000ll;
		//}
		//printf(" % lld. % 09llds\n", dsec, dnsec);


		//windows��ȷ��ʱ
		long  float head, tail, freq;//timers
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		normal_gauss(data, n);//�����㷨
		//sse_gauss(data, n);//sse�㷨
		//neon_gauss(data, n);//neon�㷨
		//avx_gauss(data, n);//AVX�㷨
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout  << (tail - head)  / freq  << endl;


		//�ͷŶ�̬�ڴ�
		for (int i = 0; i < n; i++) {
			delete[] data[i];
		}
		delete[] data;
	}
	

	return 0;
}