#include<iostream>
#include<algorithm>
#include<fstream>
#include<sstream>
#include<omp.h>
#include"mpi.h"
using namespace std;
const int col = 8399, elinenum = 4535, num_thread = 8, mpisize = 8;//列数、被消元行数
bool parallel = 1;
int isupgrade;
int tmp = 0;
const int bytenum = (col - 1) / 32 + 1;   //每个实例中的byte型数组数


class bitmatrix {
public:
	int mycol;    //首项
	int *mybyte;
	bitmatrix() {    //初始化
		mycol = -1;
		mybyte = new int[bytenum];
		for (int i = 0; i < bytenum; i++)
			mybyte[i] = 0;
	}
	bool isnull() {  //判断当前行是否为空行
		if (mycol == -1)return 1;
		return 0;
	}
	void insert(int x) { //数据读入
		if (mycol == -1)mycol = x;
		int a = x / 32, b = x % 32;
		mybyte[a] |= (1 << b);
	}
	void doxor(bitmatrix b) {  //两行做异或操作，由于结果留在本实例中，只有被消元行能执行这一操作,且异或操作后要更新首项
		for (int i = 0; i < bytenum; i++)
			mybyte[i] ^= b.mybyte[i];
		for (int i = bytenum - 1; i >= 0; i--)
			for (int j = 31; j >= 0; j--)
				if ((mybyte[i] & (1 << j)) != 0) {
					mycol = i * 32 + j;
					return;
				}
		mycol = -1;
	}
};
bitmatrix *eliminer = new bitmatrix[col], *eline = new bitmatrix[elinenum], *pass = new bitmatrix[mpisize];
void readdata() {
	ifstream ifs;
	ifs.open("D:\\VS项目\\mpi1\\x64\\Debug\\eliminer1.txt");  //消元子
	string temp;
	while (getline(ifs, temp)) {
		istringstream ss(temp);
		int x;
		int trow = 0;
		while (ss >> x) {
			if (!trow)trow = x;    //第一个读入元素代表行号
			eliminer[trow].insert(x);
		}
	}
	ifs.close();
	ifstream ifs2;
	ifs2.open("D:\\VS项目\\mpi1\\x64\\Debug\\eline1.txt");     //被消元行,读入方式与消元子不同
	int trow = 0;
	while (getline(ifs2, temp)) {
		istringstream ss(temp);
		int x;
		while (ss >> x) {
			eline[trow].insert(x);
		}
		trow++;
	}
	ifs2.close();
}
void printres() { //打印结果
	for (int i = 0; i < elinenum; i++) {
		if (eline[i].isnull()) { puts(""); continue; }   //空行的特殊情况
		for (int j = bytenum - 1; j >= 0; j--) {
			for (int k = 31; k >= 0; k--)
				if ((eline[i].mybyte[j] & (1 << k)) != 0) {     //一个错误调了半小时，谨记当首位为1时>>不等于除法！
					printf("%d ", j * 32 + k);
				}
		}
		puts("");
	}
}
/*void dowork() {  //串行消元--被消元行->消元子
	int rank;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	//获取当前进程号
	double st = MPI_Wtime();	 //计时
	for (int i = 0; i < elinenum; i++) {
		while (!eline[i].isnull()) {  //只要被消元行非空，循环处理
			int tcol = eline[i].mycol;  //被消元行的首项
			if (!eliminer[tcol].isnull())    //如果存在对应消元子
				eline[i].doxor(eliminer[tcol]);
			else {
				eliminer[tcol] = eline[i];    //由于被消元行升格为消元子后不参与后续处理，可以直接用=来浅拷贝
				break;
			}
		}
	}
	double ed = MPI_Wtime();	 //计时
	if (rank == 0) {	//只有0号进程中有最终结果
		printf("cost time:%.4lf\n", ed - st);
	}
}*/

void dowork() {  //串行消元--消元子->被消元行
	int rank;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	//获取当前进程号
	int r1 = rank * (elinenum / mpisize), r2 = (rank == mpisize - 1) ? elinenum - 1 : (rank + 1)*(elinenum / mpisize) - 1;
	double st = MPI_Wtime();	 //计时开始
	int i, j, k;
	for (i = col - 1; i >= 0; i--) {
		if (!eliminer[i].isnull()) {
			for (j = r1; j <= r2; j++) {
				if (eline[j].mycol == i)
					eline[j].doxor(eliminer[i]);
			}
		}
		else {
			//cout << rank<<" "<<2 << endl;
			isupgrade = -1;
			int t = -1;
			if (rank != 0) {
				for (k = r1; k <= r2; k++)
					if (eline[k].mycol == i) {
						pass[rank] = eline[k];
						t = k;
						MPI_Send(&t, 1, MPI_INT, 0, k + elinenum * 4 + 3, MPI_COMM_WORLD);
						MPI_Send(&pass[rank].mybyte[0], bytenum, MPI_INT, 0, k + 3, MPI_COMM_WORLD);	//各进程内消元完毕的被消元行传回0号进程
						MPI_Send(&pass[rank].mycol, 1, MPI_INT, 0, k + elinenum + 3, MPI_COMM_WORLD);
						break;
					}
				if (k > r2) {
					MPI_Send(&t, 1, MPI_INT, 0, k + elinenum * 4 + 3, MPI_COMM_WORLD);
					MPI_Send(&pass[rank].mybyte[0], bytenum, MPI_INT, 0, k + 3, MPI_COMM_WORLD);	//各进程内消元完毕的被消元行传回0号进程
					MPI_Send(&pass[rank].mycol, 1, MPI_INT, 0, k + elinenum + 3, MPI_COMM_WORLD);
				}
			}
			else {
				for (k = mpisize - 1; k > 0; k--) {
					MPI_Recv(&t, 1, MPI_INT, k, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					MPI_Recv(&pass[k].mybyte[0], bytenum, MPI_INT, k, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					MPI_Recv(&pass[k].mycol, 1, MPI_INT, k, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					if (t != -1) {
						eliminer[i] = pass[k];
						isupgrade = t;

					}
				}
				for (k = r1; k <= r2; k++)
					if (eline[k].mycol == i) {
						eliminer[i] = eline[k];
						isupgrade = k;
						break;
					}
				MPI_Send(&isupgrade, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
				if (isupgrade != -1) {
					MPI_Send(&eliminer[i].mybyte[0], bytenum, MPI_INT, 1, 1, MPI_COMM_WORLD);
					MPI_Send(&eliminer[i].mycol, 1, MPI_INT, 1, 2, MPI_COMM_WORLD);
				}
			}
			//MPI_Barrier(MPI_COMM_WORLD);
			if (rank != 0) {
				MPI_Recv(&isupgrade, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
				if (isupgrade != -1) {
					MPI_Recv(&eliminer[i].mybyte[0], bytenum, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);
					MPI_Recv(&eliminer[i].mycol, bytenum, MPI_INT, rank-1, 2, MPI_COMM_WORLD, &status);
				}
				if (rank != mpisize - 1) {
					MPI_Send(&isupgrade, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
					if (isupgrade != -1) {
						MPI_Send(&eliminer[i].mybyte[0], bytenum, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
						MPI_Send(&eliminer[i].mycol, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD);
					}
				}
			}
			if (isupgrade != -1 && r2 >= isupgrade)
				for (j = r1; j <= r2; j++) {
					if (eline[j].mycol == i && j != isupgrade)
						eline[j].doxor(eliminer[i]);
				}
		}
	}

	if (rank != 0)
		for (k = r1; k <= r2; k++) {
			MPI_Send(&eline[k].mybyte[0], bytenum, MPI_INT, 0, k + 3 + elinenum * 2, MPI_COMM_WORLD);	//各进程内消元完毕的被消元行传回0号进程
			MPI_Send(&eline[k].mycol, 1, MPI_INT, 0, k + 3 + elinenum * 3, MPI_COMM_WORLD);
		}
	else
		for (k = 1; k < mpisize; k++) {
			int t1 = k * (elinenum / mpisize), t2 = (k == mpisize - 1) ? elinenum - 1 : (k + 1)*(elinenum / mpisize) - 1;
			for (int q = t1; q <= t2; q++) {
				MPI_Recv(&eline[q].mybyte[0], bytenum, MPI_INT, k, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				MPI_Recv(&eline[q].mycol, 1, MPI_INT, k, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			}
		}
	double ed = MPI_Wtime();	 //计时结束
	if (rank == 0) {	//只有0号进程中有最终结果
		printf("cost time:%.4lf\n", ed - st);
		//printres();
	}
}

int main() {
	MPI_Init(0, 0);
	readdata();
	dowork();
	MPI_Finalize();
	return 0;
}