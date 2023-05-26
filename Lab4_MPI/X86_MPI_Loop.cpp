#include "mpi.h" 
#include <iostream> 
#include <emmintrin.h>
#include <immintrin.h>
#include<algorithm>
using namespace std;
const int mpisize = 16;
int n = 2000;
float** m1;//数据矩阵
int main(int argc, char* argv[])
{
	int m[9] = { 100,200,300,400,500,1000,2000,3000,4000 };
	for (int i = 0; i < 9; i++) {
		n = m[i];
		m1 = new float* [n];//矩阵数据

		for (int i = 0; i < n; i++) {
			m1[i] = new float[n];
		}

		float  time1;
		bool parallel = 1;

		int rank;
		double st, ed;
		MPI_Status status;
		MPI_Init(0, 0);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);	//获取当前进程号
		if (rank == 0) {	//0号进程实现初始化
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++)
					m1[i][j] = 0;
				m1[i][i] = 1.0;
				for (int j = i + 1; j < n; j++)
					m1[i][j] = rand() % 100 + 1;
			}
			for (int k = 0; k < n; k++)
				for (int i = k + 1; i < n; i++)
					for (int j = 0; j < n; j++)
						m1[i][j] = int((m1[i][j] + m1[k][j])) % 100 + 1.0;
			for (int j = 1; j < mpisize; j++) {
				for (int i = j; i < n; i += mpisize - 1)
					MPI_Send(&m1[i][0], n, MPI_FLOAT, j, i, MPI_COMM_WORLD);
			}
		}
		else
			for (int i = rank; i < n; i += mpisize - 1)
				MPI_Recv(&m1[i][0], n, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
		MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
		st = MPI_Wtime();	 //计时
		for (int k = 0; k < n; k++) {
			if (rank == 0) {	//0号进程负责除法部分
				for (int j = k + 1; j < n; j++)
					m1[k][j] /= m1[k][k];
				m1[k][k] = 1.0;
				for (int j = 1; j < mpisize; j++)
					MPI_Send(&m1[k][0], n, MPI_FLOAT, j, n + k + 1, MPI_COMM_WORLD);
			}
			else
				MPI_Recv(&m1[k][0], n, MPI_FLOAT, 0, n + k + 1, MPI_COMM_WORLD, &status);
			if (rank != 0) {
				int r2 = rank;
				while (r2 < k + 1)r2 += mpisize - 1;
				for (int i = r2; i < n; i += mpisize - 1) {//负责k+1行之后的各进程并发进行减法
					for (int j = k + 1; j < n; j++)
						m1[i][j] -= m1[i][k] * m1[k][j];
					m1[i][k] = 0;
					if (i == k + 1)
						MPI_Send(&m1[i][0], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);	//减法结果广播回0号进程
				}
			}
			else if (k < n - 1)
				MPI_Recv(&m1[k + 1][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		}
		MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
		ed = MPI_Wtime();	 //计时
		MPI_Finalize();
		if (rank == 0) {	//只有0号进程中有最终结果
			printf("cost time:%.4lf\n", ed - st);
		}

		//释放动态内存
		for (int i = 0; i < n; i++) {
			delete[] m1[i];
		}
		delete[] m1;
	}
	return 0;
}