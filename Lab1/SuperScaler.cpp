#include<iostream>
#include<Windows.h>
using namespace std;

const int shift = 27;
const int Size = 1<<shift;  // 2^n
const int RepTime = 100;

int arr[Size];
int sum1=0,sum2=0,sum3=0,sum4=0;

void Init(){
    for(int &i:arr)
        i = 1;
}

void PrintInfo(){
    cout<<"Shift:"<<shift<<endl;
    cout<<"Size:"<<Size<<endl;
    cout<<"RepeatTime:"<<RepTime<<endl;
}


void Normal(){
    for(int &i:arr)
        sum1+=i;
}

void MultipleLink(){
    int sum2_1=0,sum2_2=0;
    for(int i=0;i<Size;i+=2){
        sum2_1 += arr[i];
        sum2_2 += arr[i+1];
    }
    sum2 = sum2_1+sum2_2;
}

void Recur(int n){
    if(n == 1)
        return;
    else{
        for(int i=0;i<n/2;i++)
            arr[i] += arr[n-1-i];
        n/=2;
        Recur(n);
    }
}

void DoubleLoop(){
    for(int m=Size; m>1;m/=2)
        for(int i=0;i<m/2;i++)
            arr[i] = arr[i*2]+arr[i*2+1];
}


void Run(){
    long long head,tail,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    cout<<"Normal:"<<endl;
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int t=0;t<RepTime;t++){
//        sum1=0;
        Normal();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout<<"\tTime:"<<(tail-head)*1000.0/freq/RepTime<<"ms"<<endl;
    // ---------------------------------------------------------------------------------------------
    cout<<"MultipleLink:"<<endl;
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int t=0;t<RepTime;t++){
//        sum2=0;
        MultipleLink();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout<<"\tTime:"<<(tail-head)*1000.0/freq/RepTime<<"ms"<<endl;
    // ---------------------------------------------------------------------------------------------
    cout<<"Recur:"<<endl;
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int t=0;t<RepTime;t++){
        Recur(Size);
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout<<"\tTime:"<<(tail-head)*1000.0/freq/RepTime<<"ms"<<endl;
    // ---------------------------------------------------------------------------------------------
    cout<<"DoubleLoop:"<<endl;
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int t=0;t<RepTime;t++){
        DoubleLoop();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout<<"\tTime:"<<(tail-head)*1000.0/freq/RepTime<<"ms"<<endl;
    // ---------------------------------------------------------------------------------------------

}




int main() {

	Init();
    PrintInfo();
    Run();


	return 0;
}
