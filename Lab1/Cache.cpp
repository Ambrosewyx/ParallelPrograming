#include<iostream>
#include<windows.h>
using namespace std;

const int Max = 5000;  // Matrix Scale
const int RepTime = 100;    // Repeat Time

int matrix[Max][Max], col[Max], sum1[Max], sum2[Max];

// Initialize
void Init_Matrix_Col(){
    for(int i=0;i<Max;i++){
        col[i] = i;
        for(int j=0;j<Max;j++)
            matrix[i][j] = i+j;
    }
}

void InitSum1(){
    for(int & i : sum1)
        i=0;
}

void InitSum2(){
    for(int & i : sum2)
        i=0;
}

void Correct(){
    for(int i=0;i<Max;i++){
        if(sum1[i] != sum2[i]){
            cout<<"Correct Result:False"<<endl;
            return;
        }
    }
    cout<<"Correct Result:True"<<endl;
}

void PrintInfo(){
    Correct();
    cout<<"Martrix Scale:"<<Max<<endl;
    cout<<"Repeat Time:"<<RepTime<<endl;
}

void NormalAlgorithm(){
    long long head,tail,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    for(int t=0;t<RepTime;t++){

        InitSum1();
        for(int j=0;j<Max;j++){
            for(int i=0;i<Max;i++){
                sum1[j] += matrix[i][j]*col[i];
            }
        }

    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout<<"Time:"<<(tail-head)*1000.0/freq/RepTime<<"ms"<<endl;
}

void CacheOptimize(){
    long long head,tail,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    for(int t=0;t<RepTime;t++){

        InitSum2();
        for(int i=0;i<Max;i++){
            for(int j=0;j<Max;j++){
                sum2[j] += matrix[i][j]*col[i];
            }
        }

    }


    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout<<"Time:"<<(tail-head)*1000.0/freq/RepTime<<"ms"<<endl;
}

int main() {
    Init_Matrix_Col();

    //cout<<"NormalAlgorithm:";
    //NormalAlgorithm();

    cout<<"CacheOptimize:";
    CacheOptimize();

    PrintInfo();

	return 0;
}
