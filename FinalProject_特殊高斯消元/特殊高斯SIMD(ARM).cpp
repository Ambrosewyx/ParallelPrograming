#include<iostream>
#include<fstream>
#include<sstream>
#include <ctime>
#include <ratio>
#include <chrono>
#include<arm_neon.h>
#include<stdlib.h>
using namespace std;
const int col=130,elinenum=8; //列数、被消元行数
int bytenum=(col-1)/32+1;   //每个实例中的byte型数组数
class bitmatrix{
public:
    int mycol;    //首项
    int *mybyte;    
    bitmatrix(){    //初始化
        mycol=-1;
        mybyte=(int *)aligned_alloc(1024, 1024 * sizeof(int));
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
    void doxor(bitmatrix b){  //两行做异或操作，由于结果留在本实例中，只有被消元行能执行这一操作,且异或操作后要更新首项
        int i=0;
        for(;i+4<=bytenum;i+=4){
            int32x4_t byte1=vld1q_s32(mybyte+i);
            int32x4_t byte2=vld1q_s32(b.mybyte+i);
            byte1=veorq_s32(byte1,byte2);
            vst1q_s32(mybyte+i,byte1);
            }
        for(;i<bytenum;i++)
             mybyte[i]^=b.mybyte[i];
        for(int i=bytenum-1;i>=0;i--)
            for(int j=31;j>=0;j--)
                if((mybyte[i]&(1<<j))!=0){
                    mycol=i*32+j;
                    return;
                }
        mycol=-1;  
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
void dowork(){  //消元
    for(int i=0;i<elinenum;i++){
        while(!eline[i].isnull()){  //只要被消元行非空，循环处理
            int tcol = eline[i].mycol;  //被消元行的首项
            if(!eliminer[tcol].isnull())    //如果存在对应消元子
                eline[i].doxor(eliminer[tcol]);
            else{
                eliminer[tcol]=eline[i];    //由于被消元行升格为消元子后不参与后续处理，可以直接用=来浅拷贝
                break;
            }
        }
    }
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
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    readdata();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    std::cout<<"readtime: "<<duration_cast<duration<double>>(t2-t1).count()<<std::endl;
    dowork();
    high_resolution_clock::time_point t3 = high_resolution_clock::now();
    std::cout<<"serial: "<<duration_cast<duration<double>>(t3-t2).count()<<std::endl;
    //printres();
    //system("pause");
    return 0;
}