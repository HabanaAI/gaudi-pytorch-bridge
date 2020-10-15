
# Test for random small data
export TP_MODEL_PARAM_DUMP_ENABLE=1
export TP_MODEL_PARAM_DUMP_ITER_INDICES_TO_DUMP=0,1
if true; then

NUM_NODES=2
BATCH_SIZE=64
DATA_SIZE=`expr ${BATCH_SIZE} \\* 2`

echo ${DATA_SIZE}

LOG_FILE_PATH=`pwd`"/random_data"
SINGLE_CHIP_TP_DATA_DUMP_PATH=${LOG_FILE_PATH}"/habana1"
MULTI_CHIP_TP_DATA_DUMP_PATH=${LOG_FILE_PATH}"/habanam"

HCL_CONFIG_PATH_FILE=`pwd`"/hls${NUM_NODES}.json"
echo -en "{\n\"HCL_PORT\": 5332,\n\"HCL_TYPE\": \"HLS1\",\n\"HCL_COUNT\": ${NUM_NODES}\n}" > $HCL_CONFIG_PATH_FILE

export HCL_CONFIG_PATH="${HCL_CONFIG_PATH_FILE}"
export TP_MODEL_PARAM_DUMP_ENABLE=1


export TP_DATA_DUMP_PATH=${SINGLE_CHIP_TP_DATA_DUMP_PATH}
python -u ${PYTORCH_MODULES_ROOT_PATH}/topologies/dlrm/dlrm_s_pytorch_hpu_custom.py --arch-interaction-op=cat --arch-sparse-feature-size 16 --arch-mlp-bot 13-512-256-64-16 --arch-mlp-top 512-256-1 --mini-batch-size ${BATCH_SIZE} --learning-rate 1e-5 --data-size=${DATA_SIZE} --print-time 2>&1 |tee hpu_custom_dlrm_mini_1.log

export TP_DATA_DUMP_PATH=${MULTI_CHIP_TP_DATA_DUMP_PATH}

python -um torch.distributed.launch --nproc_per_node=${NUM_NODES} --use_env ${PYTORCH_MODULES_ROOT_PATH}/topologies/dlrm/dlrm_s_pytorch_hpu_custom.py --arch-interaction-op=cat --arch-sparse-feature-size 16 --arch-mlp-bot 13-512-256-64-16 --arch-mlp-top 512-256-1 --mini-batch-size ${BATCH_SIZE} --learning-rate 1e-5 --data-size=${DATA_SIZE} --print-time --distributed 2>&1 |tee hpu_custom_dlrm_mini_2.log

python -u ${PYTORCH_MODULES_ROOT_PATH}/topologies/dlrm/merge_log_files.py --ip-path ${MULTI_CHIP_TP_DATA_DUMP_PATH} --world-size ${NUM_NODES}

pushd ${PYTORCH_MODULES_ROOT_PATH}/topologies
python tools/scr_tensor_comparison.py --data-path1=${LOG_FILE_PATH} --device1 habanam --device2 habana1
popd

fi

# Test with medium config random data
if false; then

NUM_NODES=8
BATCH_SIZE=512
DATA_SIZE=`expr ${BATCH_SIZE} \\* 2`
LOG_FILE_PATH=`pwd`"/medium_data"
SINGLE_CHIP_TP_DATA_DUMP_PATH=${LOG_FILE_PATH}"/habana1"
MULTI_CHIP_TP_DATA_DUMP_PATH=${LOG_FILE_PATH}"/habanam"


HCL_CONFIG_PATH_FILE=`pwd`"/hls${NUM_NODES}.json"
echo -en "{\n\"HCL_PORT\": 5332,\n\"HCL_TYPE\": \"HLS1\",\n\"HCL_COUNT\": ${NUM_NODES}\n}" > $HCL_CONFIG_PATH_FILE

export HCL_CONFIG_PATH="${HCL_CONFIG_PATH_FILE}"

ARCH_EMBEDDING_SIZE=3000000-3000000-3000000-3000000-3000000-3000000-3000000-3000000
ARCH_MLP_BOTTOM=1024-1024-1024-64
ARCH_MLP_TOP=4096-4096-4096-4096-4096-4096-4096-1
export TP_DATA_DUMP_PATH=${SINGLE_CHIP_TP_DATA_DUMP_PATH}
python -u ${PYTORCH_MODULES_ROOT_PATH}/topologies/dlrm/dlrm_s_pytorch_hpu_custom.py --arch-interaction-op=cat --arch-sparse-feature-size 64 --arch-mlp-bot ${ARCH_MLP_BOTTOM} --arch-embedding-size ${ARCH_EMBEDDING_SIZE} --arch-mlp-bot ${ARCH_MLP_BOTTOM} --arch-mlp-top ${ARCH_MLP_TOP} --num-indices-per-lookup 38 --mini-batch-size ${BATCH_SIZE} --learning-rate 1e-5 --data-size=${DATA_SIZE} --print-time 2>&1 |tee hpu_custom_dlrm_medium_1.log

export TP_DATA_DUMP_PATH=${MULTI_CHIP_TP_DATA_DUMP_PATH}

python -um torch.distributed.launch --nproc_per_node=${NUM_NODES} --use_env ${PYTORCH_MODULES_ROOT_PATH}/topologies/dlrm/dlrm_s_pytorch_hpu_custom.py --arch-interaction-op=cat --arch-sparse-feature-size 64 --arch-embedding-size ${ARCH_EMBEDDING_SIZE} --arch-mlp-bot ${ARCH_MLP_BOTTOM} --arch-mlp-top ${ARCH_MLP_TOP} --num-indices-per-lookup 38 --mini-batch-size ${BATCH_SIZE} --learning-rate 1e-5 --data-size=${DATA_SIZE} --print-time --distributed 2>&1 |tee hpu_custom_dlrm_medium_8.log

python -u ${PYTORCH_MODULES_ROOT_PATH}/topologies/dlrm/merge_log_files.py --ip-path ${MULTI_CHIP_TP_DATA_DUMP_PATH} --world-size ${NUM_NODES} --arch-embedding-size ${ARCH_EMBEDDING_SIZE}

pushd ${PYTORCH_MODULES_ROOT_PATH}/topologies
python tools/scr_tensor_comparison.py --data-path1=${LOG_FILE_PATH} --device1 habanam --device2 habana1
popd

fi

if true; then
NUM_NODES=8
BATCH_SIZE=512
NUM_BATCHES=2

LOG_FILE_PATH=`pwd`"/vanilla_kaggle_data"
SINGLE_CHIP_TP_DATA_DUMP_PATH=${LOG_FILE_PATH}"/habana1"
MULTI_CHIP_TP_DATA_DUMP_PATH=${LOG_FILE_PATH}"/habanam"

HCL_CONFIG_PATH_FILE=`pwd`"/hls${NUM_NODES}.json"
echo -en "{\n\"HCL_PORT\": 5332,\n\"HCL_TYPE\": \"HLS1\",\n\"HCL_COUNT\": ${NUM_NODES}\n}" > $HCL_CONFIG_PATH_FILE

export HCL_CONFIG_PATH="${HCL_CONFIG_PATH_FILE}"

export TP_DATA_DUMP_PATH=${SINGLE_CHIP_TP_DATA_DUMP_PATH}
ARCH_EMBEDDING_SIZE=1460-583-10131227-2202608-305-24-12517-633-3-93145-5683-8351593-3194-27-14992-5461306-10-5652-2173-4-7046547-18-15-286181-105-142572
ARCH_MLP_BOTTOM=13-512-256-64-16
ARCH_MLP_TOP=512-256-1

python -u ${PYTORCH_MODULES_ROOT_PATH}/topologies/dlrm/dlrm_s_pytorch_hpu_custom.py --data-generation=dataset --data-set=kaggle --raw-data-file="/software/data/pytorch/kaggle_data/train.txt" --processed-data-file="/software/data/pytorch/kaggle_data//kaggleAdDisplayChallenge_processed.npz" --arch-interaction-op=cat --arch-sparse-feature-size=16 --arch-mlp-bot=${ARCH_MLP_BOTTOM} --arch-mlp-top=${ARCH_MLP_TOP} --arch-embedding-size=${ARCH_EMBEDDING_SIZE} --num-indices-per-lookup 1 --mini-batch-size ${BATCH_SIZE} --num-batches=${NUM_BATCHES} --learning-rate 1e-5 --print-time 2>&1 | tee hpu_custom_dlrm_vanilla_1.log


export TP_DATA_DUMP_PATH=${MULTI_CHIP_TP_DATA_DUMP_PATH}

python -um torch.distributed.launch --nproc_per_node=${NUM_NODES} --use_env ${PYTORCH_MODULES_ROOT_PATH}/topologies/dlrm/dlrm_s_pytorch_hpu_custom.py --data-generation=dataset --data-set=kaggle --raw-data-file="/software/data/pytorch/kaggle_data/train.txt" --processed-data-file="/software/data/pytorch/kaggle_data//kaggleAdDisplayChallenge_processed.npz" --arch-interaction-op=cat --arch-sparse-feature-size=16 --arch-mlp-bot=${ARCH_MLP_BOTTOM} --arch-mlp-top=${ARCH_MLP_TOP} --arch-embedding-size=${ARCH_EMBEDDING_SIZE} --num-indices-per-lookup 1 --mini-batch-size ${BATCH_SIZE} --num-batches=${NUM_BATCHES} --learning-rate 1e-5 --print-time --distributed 2>&1 |tee hpu_custom_dlrm_vanilla_8.log

python -u ${PYTORCH_MODULES_ROOT_PATH}/topologies/dlrm/merge_log_files.py --ip-path ${MULTI_CHIP_TP_DATA_DUMP_PATH} --world-size ${NUM_NODES} --arch-embedding-size ${ARCH_EMBEDDING_SIZE}

pushd ${PYTORCH_MODULES_ROOT_PATH}/topologies
python tools/scr_tensor_comparison.py --data-path1=${LOG_FILE_PATH} --device1 habanam --device2 habana1
popd
fi

