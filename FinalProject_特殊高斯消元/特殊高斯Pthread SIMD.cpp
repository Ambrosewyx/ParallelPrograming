#include<iostream>
#include<fstream>
#include<sstream>
#include <ctime>
#include <ratio>
#include <chrono>
#include<pthread.h>
#include<semaphore.h>
#include <emmintrin.h>
#include <immintrin.h>
using namespace std;
const int col=3799,elinenum=1953,num_thread=8; //列数、被消元行数
int tmp=0;
int bytenum=(col-1)/32+1;   //每个实例中的byte型数组数
 typedef struct{
    int t_id;   //线程id
}threadparam_t;
pthread_barrier_t barrier;
pthread_barrier_t barrier2;
class bitmatrix{
public:
    int mycol;    //首项
    int *mybyte;    
    bitmatrix(){    //初始化
        mycol=-1;
        mybyte = (int *)aligned_alloc(1024, 1024 * sizeof(int));;
        for(int i=0;i<bytenum;i++)
            mybyte[i]=0; 
    }
    bool isnull(){  //判断当前行是否为空行
        if(mycol==-1)return 1;
        return 0;
    }   
    void insert(int x){ //数据读入
        if(mycol==-1)mycol=x;
        int a=x/32,b=x%32;
        mybyte[a]|=(1<<b);
    }
};
bitmatrix *eliminer=new bitmatrix[col],*eline=new bitmatrix[elinenum];
void readdata(){
    ifstream ifs;
    ifs.open("eliminer1.txt");  //消元子
    string temp;
    while(getline(ifs,temp)){
        istringstream ss(temp);
        int x;
        int trow=0;
        while(ss>>x){
            if(!trow)trow=x;    //第一个读入元素代表行号
            eliminer[trow].insert(x);
        }
    }
    ifs.close();
    ifstream ifs2;
    ifs2.open("eline1.txt");     //被消元行,读入方式与消元子不同
    int trow=0;
    while(getline(ifs2,temp)){
        istringstream ss(temp);
        int x;
        while(ss>>x){
            eline[trow].insert(x);
        }
        trow++;
    }
    ifs2.close();
}
void *threadfunc(void *param){
    threadparam_t *p=(threadparam_t *)param;
    int t_id=p->t_id;
   for(int i=col-1;i>=0;i--) 
        if(!eliminer[i].isnull()){
            for(int j=t_id;j<elinenum;j+=num_thread){
                if(eline[j].mycol==i){
                    int k=0;
                    for(;k+4<bytenum;k+=4){
                      __m128i byte1=_mm_load_si128((__m128i*)(eline[j].mybyte+k));     //错误修正：它竟然要加括号
                      __m128i byte2=_mm_load_si128((__m128i*)(eliminer[i].mybyte+k));
                      byte1=_mm_xor_si128(byte1,byte2);
                      _mm_store_si128((__m128i*)(eline[j].mybyte+k),byte1);
                    }
                    for(;k<bytenum;k++)
                   eline[j].mybyte[k]^=eliminer[i].mybyte[k];
                    bool f=1;
                    for(int p=bytenum-1;p>=0&&f;p--)
                        for(int k=31;k>=0&&f;k--)
                            if((eline[j].mybyte[p]&(1<<k))!=0){
                                eline[j].mycol=p*32+k;
                                f=0;
                            }
                    if(f)eline[j].mycol=-1;
                    }
            }
            }
        else{
            pthread_barrier_wait(&barrier);
            if(t_id==0)
                for(int j=0;j<elinenum;j++){
                    if(eline[j].mycol==i){
                        eliminer[i]=eline[j];
                        tmp=j+1;
                        break;
                        }
                    tmp=j+2;
                }
            pthread_barrier_wait(&barrier2);
            int temp=t_id;
            while(temp<tmp)temp+=num_thread;
            for(int j=temp;j<elinenum;j+=num_thread){
                if(eline[j].mycol==i){
                   int k=0;
                    for(;k+4<bytenum;k+=4){
                      __m128i byte1=_mm_load_si128((__m128i*)(eline[j].mybyte+k));     //错误修正：它竟然要加括号
                      __m128i byte2=_mm_load_si128((__m128i*)(eliminer[i].mybyte+k));
                      byte1=_mm_xor_si128(byte1,byte2);
                      _mm_store_si128((__m128i*)(eline[j].mybyte+k),byte1);
                    }
                    for(;k<bytenum;k++)
                   eline[j].mybyte[k]^=eliminer[i].mybyte[k];
                    bool f=1;
                    for(int p=bytenum-1;p>=0&&f;p--)
                        for(int k=31;k>=0&&f;k--)
                            if((eline[j].mybyte[p]&(1<<k))!=0){
                                eline[j].mycol=p*32+k;
                                f=0;
                            }
                    if(f)eline[j].mycol=-1;
                    }
            }
            }
    pthread_exit(NULL);
}
void dowork(){  //消元
    pthread_barrier_init(&barrier,NULL,num_thread);
    pthread_barrier_init(&barrier2,NULL,num_thread);
    //创建线程
    pthread_t handles[num_thread];
    threadparam_t param[num_thread];
    for(int t_id=0;t_id<num_thread;t_id++){
        param[t_id].t_id=t_id;
        pthread_create(&handles[t_id],NULL,threadfunc,(void*)&param[t_id]);
    }
    for(int i=0;i<num_thread;i++)
        pthread_join(handles[i],NULL);
    pthread_barrier_destroy(&barrier);
    pthread_barrier_destroy(&barrier2);
}
void printres(){ //打印结果
    for(int i=0;i<elinenum;i++){
        if(eline[i].isnull()){puts("");continue;}   //空行的特殊情况
        for(int j=bytenum-1;j>=0;j--){
            for(int k=31;k>=0;k--)
                if((eline[i].mybyte[j]&(1<<k))!=0){     //一个错误调了半小时，谨记当首位为1时>>不等于除法！
                    printf("%d ",j*32+k);
                }
                }
        puts("");
    }
}
int main(){
    readdata();
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    dowork();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    std::cout<<"serial: "<<duration_cast<duration<double>>(t2-t1).count()<<std::endl;
    //printres();
    //system("pause");
    return 0;
}