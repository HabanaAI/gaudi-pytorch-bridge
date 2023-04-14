#!/bin/bash


#export PT_ENABLE_HABANA_STREAMASYNC=0
#export GRAPH_VISUALIZATION=1
#export LOG_LEVEL_ALL_PT=0
#export ENABLE_CONSOLE=true

TEST_FILE_PATH=$PYTORCH_MODULES_ROOT_PATH/tests/fused_ops/flash_attention

PERF_MEM_TEST_LOG_DIR=flash_perf_mem_test_log
PERF_CYC_TEST_LOG_DIR=flash_perf_cyc_test_log
ACC_TEST_LOG_DIR=flash_acc_test_log

perf_test_mem() {
echo " Running Flash attention Memory Perf Tests "
rm -rf $PERF_MEM_TEST_LOG_DIR
mkdir -p $PERF_MEM_TEST_LOG_DIR

rm  habana_log.livealloc.log_0

export PT_HPU_POOL_MEM_ENABLE_TENSOR_INFO=1
export PT_HABANA_MEM_LOG_LEVEL=5

#FLASH_ATTN_ALGO_PAPER=1 FLASH_ATTN_DBG_PERF_CMP_RUN_FLASH=1 FLASH_ATTN_ALGO_CFG_TR=4 FLASH_ATTN_ALGO_CFG_TC=4 python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $PERF_MEM_TEST_LOG_DIR/pt_log_flash_attn_paper_tr4_tc4.txt
#mv habana_log.livealloc.log_0 $PERF_MEM_TEST_LOG_DIR/live_alloc_log_flash_paper_tr4_tc4.txt

FLASH_ATTN_ALGO_NO_SLICE_SOFTMAX_NORNG=1 FLASH_ATTN_DBG_PERF_CMP_RUN_FLASH=1 python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $PERF_MEM_TEST_LOG_DIR/pt_log_flash_attn_no_slice_softmax_norng.txt
mv habana_log.livealloc.log_0 $PERF_MEM_TEST_LOG_DIR/live_alloc_log_flash_no_slice_softamx_norng.txt
#mv .graph_dumps $PERF_MEM_TEST_LOG_DIR/gr_flash_no_slice_softamx_norng

#FLASH_ATTN_ALGO_Q_SLICE=1 FLASH_ATTN_DBG_PERF_CMP_RUN_FLASH=1 FLASH_ATTN_ALGO_CFG_TR=4  python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $PERF_MEM_TEST_LOG_DIR/pt_log_flash_attn_q_slice_tr4.txt
#mv habana_log.livealloc.log_0 $PERF_MEM_TEST_LOG_DIR/live_alloc_log_flash_q_slice_tr4.txt
#mv .graph_dumps gr_dump_flash_q_slice_tr4

#FLASH_ATTN_ALGO_Q_SLICE=1 FLASH_ATTN_DBG_PERF_CMP_RUN_FLASH=1 FLASH_ATTN_ALGO_CFG_TR=8  python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $PERF_MEM_TEST_LOG_DIR/pt_log_flash_attn_q_slice_tr8.txt
#mv habana_log.livealloc.log_0 $PERF_MEM_TEST_LOG_DIR/live_alloc_log_flash_q_slice_tr8.txt
#mv .graph_dumps gr_dump_flash_q_slice_tr8

FLASH_ATTN_DBG_PERF_CMP_RUN_VANILLA=1 python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $PERF_MEM_TEST_LOG_DIR/pt_log_vanilla_attn.txt
mv habana_log.livealloc.log_0 $PERF_MEM_TEST_LOG_DIR/live_alloc_log_vanilla.txt
#mv .graph_dumps $PERF_MEM_TEST_LOG_DIR/gr_vanilla
}

perf_test_cycles() {
echo " Running Flash attention Cycles Perf Tests "
mkdir -p $PERF_CYC_TEST_LOG_DIR
export FLASH_ATTN_DBG_PERF_CMP_RUN_CYCLES=1
export HABANA_PROFILE=1
FLASH_ATTN_ALGO_NO_SLICE=1 FLASH_ATTN_DBG_PERF_CMP_RUN_FLASH=1 python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $PERF_CYC_TEST_LOG_DIR/pt_log_flash_attn_no_slice.txt

FLASH_ATTN_DBG_PERF_CMP_RUN_VANILLA=1 python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $PERF_CYC_TEST_LOG_DIR/pt_log_vanilla_attn.txt

}
acc_test() {

echo " Running Flash attention Accuracy Tests "
mkdir -p $ACC_TEST_LOG_DIR

#FLASH_ATTN_ALGO_PAPER=1 FLASH_ATTN_DBG_USE_DROPOUT_STUB=1 FLASH_ATTN_ALGO_CFG_TR=4 FLASH_ATTN_ALGO_CFG_TC=4 python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $ACC_TEST_LOG_DIR/pt_log_flash_attn_paper_tr4_tc4.txt

FLASH_ATTN_ALGO_NO_SLICE=1 FLASH_ATTN_DBG_USE_DROPOUT_STUB=1 python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $ACC_TEST_LOG_DIR/pt_log_flash_attn_no_slice.txt
FLASH_ATTN_ALGO_NO_SLICE_SOFTMAX=1 FLASH_ATTN_DBG_USE_DROPOUT_STUB=1 python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $ACC_TEST_LOG_DIR/pt_log_flash_attn_no_slice_softmax.txt
#FLASH_ATTN_ALGO_NO_SLICE_SOFTMAX_NORNG=1 FLASH_ATTN_DBG_USE_DROPOUT_STUB=1 python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $ACC_TEST_LOG_DIR/pt_log_flash_attn_no_slice_softmax_norng.txt


FLASH_ATTN_ALGO_Q_SLICE=1 FLASH_ATTN_DBG_USE_DROPOUT_STUB=1 FLASH_ATTN_ALGO_CFG_TR=4  python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $ACC_TEST_LOG_DIR/pt_log_flash_attn_q_slice_tr4.txt

#FLASH_ATTN_ALGO_Q_SLICE=1 FLASH_ATTN_DBG_USE_DROPOUT_STUB=1 FLASH_ATTN_ALGO_CFG_TR=8  python $TEST_FILE_PATH/test_flash_attention.py 2>&1 | tee $ACC_TEST_LOG_DIR/pt_log_flash_attn_q_slice_tr8.txt
}

acc_test
#perf_test_cycles
#perf_test_mem
