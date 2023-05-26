#include "mpi.h" 
#include <iostream> 
#include <emmintrin.h>
#include <immintrin.h>
#include<algorithm>
using namespace std;
const int mpisize = 8;
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
		int r1 = rank * (n / mpisize), r2 = (rank == mpisize - 1) ? n - 1 : (rank + 1) * (n / mpisize) - 1;
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
				int t1 = j * (n / mpisize), t2 = (j == mpisize - 1) ? n - 1 : (j + 1) * (n / mpisize) - 1;
				MPI_Send(&m1[t1][0], n * (t2 - t1 + 1), MPI_FLOAT, j, n + 1, MPI_COMM_WORLD);
			}
		}
		else
			MPI_Recv(&m1[r1][0], n * (r2 - r1 + 1), MPI_FLOAT, 0, n + 1, MPI_COMM_WORLD, &status);
		MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
		st = MPI_Wtime();	 //计时
		for (int k = 0; k < n; k++) {
			if (r1 <= k && r2 >= k) {		//本进程负责除法，除法完后广播给下一进程，广播完消去
				for (int j = k + 1; j < n; j++)		//除法
					m1[k][j] /= m1[k][k];
				m1[k][k] = 1.0;
				//for (int j = 0; j < n; j++)
				//	cout << m1[k][j] << " ";
				//cout << endl;
				if (rank != mpisize - 1)
					MPI_Send(&m1[k][0], n, MPI_FLOAT, rank + 1, k, MPI_COMM_WORLD);
				for (int i = k + 1; i <= r2; i++) {
					for (int j = k + 1; j < n; j++)
						m1[i][j] -= m1[i][k] * m1[k][j];
					m1[i][k] = 0;
				}
			}
			else if (r1 > k) {
				MPI_Recv(&m1[k][0], n, MPI_FLOAT, rank - 1, k, MPI_COMM_WORLD, &status);
				if (rank != mpisize - 1)
					MPI_Send(&m1[k][0], n, MPI_FLOAT, rank + 1, k, MPI_COMM_WORLD);
				for (int i = r1; i <= r2; i++) {
					for (int j = k + 1; j < n; j++)
						m1[i][j] -= m1[i][k] * m1[k][j];
					m1[i][k] = 0;
				}
			}
			//MPI_Barrier(MPI_COMM_WORLD);
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