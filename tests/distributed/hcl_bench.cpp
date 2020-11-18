#include <linux/types.h>
#include <algorithm>
#include <thread>
#include <future>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <sys/time.h>

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <bits/stdc++.h>
#include "synapse_api.h"
#include "hcl_api.h"
#include <sys/time.h>
#include <condition_variable>
#include "synapse_api.h"
#include "osal.hpp"

#include "hcl/infra/hcl_command_submission.h"
#include "hcl_api.h"
#include "hcl/hcl_dma_utils.h"
#include "hcl/hcl_channel.h"
#include "hcl/hcl_writer.h"
#include "hcl/hcl_collective_routines_tpc_reduction.h"
#include "hcl/hcl_collective_routines_local_heap.h"
#include "hcl/hcl_collective_routines_tcp.h"
#include "hcl/hcl_rank_master.h"
#include "infra/global_conf.h"


uint64_t GetMicrosec() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count());
}



using namespace std;


class SynHCLBenchTest
{
public:
    uint32_t m_DeviceId;
    int      m_fd = -1;

    void SetUp(int id)
    {
        VERIFY(synInitialize() ==  synSuccess);

        synStatus ret;
        ret = synDeviceAcquireByModuleId(&m_DeviceId, id);
        if (synSuccess != ret)
        {
            std::cout<<"Called synDeviceAcquireByDeviceType";
            ret = synDeviceAcquireByDeviceType(&m_DeviceId, synDeviceGaudi);
        }

        VERIFY(synSuccess == ret);
        m_fd = OSAL::getInstance().getFd(m_DeviceId);
    }

    void TearDown()
    {
        synDestroy();
    }

    void Send_Recv_speed(int,int,int);
    void Allreduce_speed(int,int,int);
    void Allbroadcast_speed(int,int,int);
    void AlltoAll_speed(int ,int ,int );

};

void send_recv(uint32_t m_DeviceId,int length1,int startlength,int steps,int m_fd)
    {
        int  length;
        uint64_t sendBuff, recvBuff;
        uint64_t total_length = pow(2,length1);
        VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, total_length, 0, 0, &sendBuff));
        VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, total_length, 0, 0, &recvBuff));

        iMemsetToDevice<int>(m_fd, sendBuff, 1, total_length/4);
        iMemsetToDevice<int>(m_fd, recvBuff, 0, total_length/4);

        HCL_Request hcl_request;
        printf("\nlength\ttime/message (sec)\ttransfer rate (M byte/sec)\n");
        double convert = (1000000.0/1024.0)/1024.0;
        for (length = startlength; length <=  length1; length += 1){
                uint64_t num_bytes = pow(2,length);
                int64_t start = GetMicrosec();
                for (int i = 1; i <= steps; i++){
                    HCL_Send_Tag(sendBuff,num_bytes,1,123);
                    HCL_Receive_Tag(recvBuff, num_bytes,1,124);
               }
               int64_t time_taken =  GetMicrosec()-start;

               printf("2^%d\t\t%.4f\t\t\t%.4lf\n",length,(time_taken/(double)steps)/1000000,((2.0*steps* num_bytes)/(double)time_taken)*convert);
        }
    }

void recv_send(uint32_t m_DeviceId,int length1,int startlength,int steps,int m_fd)
    {
        int  length;
        uint64_t sendBuff, recvBuff;
        uint64_t total_length = pow(2,length1);
        VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, total_length, 0, 0, &sendBuff));
        VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, total_length, 0, 0, &recvBuff));
        iMemsetToDevice<int>(m_fd, sendBuff, 1, total_length/4);
        iMemsetToDevice<int>(m_fd, recvBuff, 0, total_length/4);
        HCL_Request hcl_request;
        for (length = startlength; length <=  length1; length += 1){
                uint64_t num_bytes = pow(2,length);
                for (int i = 1; i <= steps; i++){
                    HCL_Receive_Tag(recvBuff,num_bytes ,0,123);
                    HCL_Send_Tag(sendBuff, num_bytes ,0,124);
               }

        }
    }



void SynHCLBenchTest::Send_Recv_speed(int length,int startlength,int steps)
{
    if (!isEnvExist("JSON")) return;
    HCL_Init(m_DeviceId, std::getenv("JSON"));

    uint16_t myRank;
    int comSize;
    HCL_Comm_Rank("", &myRank);
    HCL_Comm_Size("",  &comSize);
     if (myRank == 0)
     {
        send_recv(m_DeviceId,length,startlength,steps,m_fd);
        HCL_Sync("",1111);
     }
     else if (myRank == 1)
     {  recv_send(m_DeviceId,length,startlength,steps,m_fd);
        HCL_Sync("",1111);
     }
     else
     {
         HCL_Sync("",1111);
     }

    HCL_Destroy();
}

void SynHCLBenchTest::Allreduce_speed(int length1,int startlength,int steps)
{
    if (!isEnvExist("JSON")) return;
    HCL_Init(m_DeviceId, std::getenv("JSON"));
    uint16_t myRank;
    int comSize;
    HCL_Comm_Rank("", &myRank);
    HCL_Comm_Size("",  &comSize);
    uint64_t total_length = pow(2,length1);
    int  length;
    uint64_t sendBuff, recvBuff,device_intermediate_buffer;
    uint64_t intermediate_size;
    uint64_t count = total_length/4;
    HCL_Get_Intermediate_Buffer_size(&intermediate_size,
                                                  eHCLAllReduce,
                                                  count,
                                                  syn_type_single,
                                                  "");

    VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, total_length, 0, 0, &sendBuff));
    VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, total_length, 0, 0, &recvBuff));
    VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, intermediate_size, 0, 0, &device_intermediate_buffer));
    iMemsetToDevice<float>(m_fd, sendBuff, 1, total_length/4);
    iMemsetToDevice<float>(m_fd, recvBuff, 0, total_length/4);
    if(myRank == 0)
    printf("\nlength   time/message (sec)    transfer rate (M byte/sec)\n");
    HCL_Sync("", 555);
    double convert = (1000000.0/1024.0)/1024.0;
    for (length = startlength; length <=  length1; length += 1){
            uint64_t num_bytes = pow(2,length);
            count = num_bytes/4;
            int64_t start = GetMicrosec();
            for (int i = 1; i <= steps; i++){
                 HCL_Allreduce(nullptr,sendBuff,recvBuff,
                                           count,syn_type_single,
                                           device_intermediate_buffer,intermediate_size,
                                           eHCLSum,"", 1);
                                       }
            int64_t time_taken =  GetMicrosec()-start;
            if(myRank == 0)
               printf("2^%d\t\t%.4f\t\t\t%.4lf\n",length,(time_taken/(double)steps)/1000000,((steps* num_bytes)/(double)time_taken)*convert);
        }
    HCL_Destroy();
}


void SynHCLBenchTest::AlltoAll_speed(int length1,int startlength,int steps)
{
    if (!isEnvExist("JSON")) return;
    HCL_Init(m_DeviceId, std::getenv("JSON"));
    uint16_t myRank;
    int comSize;
    HCL_Comm_Rank("", &myRank);
    HCL_Comm_Size("",  &comSize);
    uint64_t total_length = pow(2,length1);
    int  length;
    uint64_t sendBuff, recvBuff,device_intermediate_buffer;
    uint64_t intermediate_size;
    uint64_t count = total_length/4;
    HCL_Get_Intermediate_Buffer_size(&intermediate_size,
                                                  eHCLAllReduce,
                                                  count,
                                                  syn_type_single,
                                                  "");

    VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, total_length, 0, 0, &sendBuff));
    VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, total_length, 0, 0, &recvBuff));
    VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, intermediate_size, 0, 0, &device_intermediate_buffer));
    iMemsetToDevice<float>(m_fd, sendBuff, 1, total_length/4);
    iMemsetToDevice<float>(m_fd, recvBuff, 0, total_length/4);
    if(myRank == 0)
    printf("\nlength   time/message (sec)    transfer rate (M byte/sec)\n");
    HCL_Sync("", 555);
    double convert = (1000000.0/1024.0)/1024.0;
    for (length = startlength; length <=  length1; length += 1){
            uint64_t num_bytes = pow(2,length);
            count = num_bytes/4;
            int64_t start = GetMicrosec();
            for (int i = 1; i <= steps; i++){
                 HCL_AlltoAll(nullptr,sendBuff,recvBuff,
                                           count,syn_type_single,
                                          device_intermediate_buffer,intermediate_size,
                                           "", 1);
                                       }
            int64_t time_taken =  GetMicrosec()-start;
            if(myRank == 0)
               printf("2^%d\t\t%.4f\t\t\t%.4lf\n",length,(time_taken/(double)steps)/1000000,((steps* num_bytes)/(double)time_taken)*convert);
        }
    HCL_Destroy();
}



void SynHCLBenchTest::Allbroadcast_speed(int length1,int startlength,int steps)
{
    if (!isEnvExist("JSON")) return;
    HCL_Init(m_DeviceId, std::getenv("JSON"));
    uint16_t myRank;
    int comSize;
    HCL_Comm_Rank("", &myRank);
    HCL_Comm_Size("",  &comSize);
    uint64_t total_length = pow(2,length1);
    int  length;
    uint64_t sendBuff, recvBuff;
    uint64_t count = total_length/4;

    VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, total_length, 0, 0, &sendBuff));
    VERIFY(synSuccess == synDeviceMalloc(m_DeviceId, total_length, 0, 0, &recvBuff));
    iMemsetToDevice<float>(m_fd, sendBuff, 1, total_length/4);
    iMemsetToDevice<float>(m_fd, recvBuff, 0, total_length/4);

    if(myRank == 0)
    printf("\nlength   time/message (sec)    transfer rate (M byte/sec)\n");
    HCL_Sync("", 555);
    double convert = (1000000.0/1024.0)/1024.0;
    for (length = startlength; length <=  length1; length += 1){
            uint64_t num_bytes = pow(2,length);
            count = num_bytes/4;
            int64_t start = GetMicrosec();
            for (int i = 1; i <= steps; i++){
                 HCL_Bcast(nullptr,sendBuff,recvBuff,
                                           count,syn_type_single,
                                           0,"", 0);
                                       }
            int64_t time_taken =  GetMicrosec()-start;
            if(myRank == 0)
               printf("2^%d\t\t%.4f\t\t\t%.4lf\n",length,(time_taken/(double)steps)/1000000,((steps* num_bytes)/(double)time_taken)*convert);
        }
    HCL_Destroy();
}




int main(int argc, char **argv)
{
    //args.n + " " + args.test + " " + args.size + " " + args.startsize +" "+args.s ;
    SynHCLBenchTest test;
    char *env_rank = std::getenv("ID");

    if(env_rank != nullptr)
    {
        int id = std::stoi(env_rank);
        std::cout.flush();
        test.SetUp(id);
        int length = std::stoi(argv[3]);
        int startlength = std::stoi(argv[4]);
        int steps = std::stoi(argv[5]);


        if(strcmp(argv[2],"sendrecv") == 0)
        {
            if(id == 0)
                {
                    printf("HCL Send_Recv Test");
                }

            test.Send_Recv_speed(length,startlength,steps);
        }
        else if(strcmp(argv[2],"allreduce") == 0)
        {
            if(id == 0)
                {
                    printf("HCL AllReduce Test <float>");
                }

            test.Allreduce_speed(length,startlength,steps);
        }
        else if(strcmp(argv[2],"broadcast") == 0)
        {
            if(id == 0)
                {
                    printf("HCL Broadcast Test <float>");
                }
            test.Allbroadcast_speed(length,startlength,steps);
        }
        else if(strcmp(argv[2],"alltoall") == 0)
        {
            if(id == 0)
                {
                    printf("HCL AlltoAll Test <float>");
                }
            test.AlltoAll_speed(length,startlength,steps);
        }

        test.TearDown();
    }
}


