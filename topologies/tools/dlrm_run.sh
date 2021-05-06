#!/bin/bash


#Script to initiate a multi node run. Set the nodes and pernode variables accordingly

#This script should be run from the nfs mounted drive which is available on all the
#compute nodes or exact copy of the script should be available on both nodes at the same path.
#The compute nodes should be able to do password less ssh to each other.

#Can run from your nfs home or create a folder in /software.. and give the right permissions and
#should be able to run it.

#command to run
#mpirun --mca btl_tcp_if_include 10.12.108.0/24 -H 10.12.108.125:8,10.12.108.88:8  -npernode 1 bash  dlrm_run.sh

nodes=2
pernode=8


if [[ -z $nodes ]] || [[ -z $pernode ]]
then
echo "Set the variables nodes and pernode"
exit -1
fi




pwdi=`pwd`

#Source the appropriate environment

. $HOME/trees/npu-stack/automation/habana_scripts/habana_env

export OPAL_PREFIX=/usr/

cd $pwdi
pwdi=`pwd`
echo $pwdi

cp $PYTORCH_MODULES_ROOT_PATH/topologies/tools/

#Get the host rank
rank=`python -c "import mpi_utils; print(mpi_utils.get_my_rank())"`

#get the rank of the root=0
root_rank_ip=`python -c "import mpi_utils; print(mpi_utils.get_root_ip())"`

if [ -z "$rank" ]
then
	echo "Error in fetching the rank"
fi

if [ -z "$root_rank_ip" ]
then
	echo "Error in fetching the root ip"
fi


config_file=`python -c "import mpi_utils; print(mpi_utils.get_hcl_config(${pernode}))"`
HCL_CONFIG_PATH_FILE=`pwd`"/hls${pernode}.json"

if [[ "$root_rank_ip" == 0 ]] && [[ -f "$HCL_CONFIG_PATH_FILE" ]]; then
rm $HCL_CONFIG_PATH_FILE
fi


echo $config_file > $HCL_CONFIG_PATH_FILE

export HCL_CONFIG_PATH="${HCL_CONFIG_PATH_FILE}"
export HABANA_HCL_STREAM_ENABLE=0
export DATACHUNK_MIN_FREE_CACHE_AMOUNT_UPPER_CP=4096
export GCFG_MIN_CSDC_FOR_COMPL_CHK=200
export DATACHUNK_SINGLE_CHUNK_SIZE_UPPER_CP=1024
export RECIPE_CACHE_BLOCK_SIZE=2048
export RECIPE_CACHE_SIZE=2048000
export HABANA_GRAPH_FUSION_OPS_FILE=$PYTORCH_MODULES_ROOT_PATH/topologies/configs/DLRM_Fusion_Ops.txt
export PT_HPU_GRAPH_FUSION_OPS_FILE=$PYTORCH_MODULES_ROOT_PATH/topologies/configs/DLRM_Fusion_Ops.txt
export PT_HABANA_LOG_MOD_MASK=0
export PT_HABANA_LOG_TYPE_MASK=0


if [ "$rank" == "0" ]
then
cat $HCL_CONFIG_PATH_FILE
fi

#edit the command appropriately its set to run for 16x
echo $PYTORCH_MODULES_ROOT_PATH

python -um torch.distributed.launch --nproc_per_node=$pernode --nnodes=$nodes --node_rank=$rank  --master_addr=$root_rank_ip  --master_port=1234 \
	--use_env  $PYTORCH_MODULES_ROOT_PATH/topologies/dlrm/dlrm_s_pytorch_hpu_graph.py --arch-interaction-op=cat --arch-sparse-feature-size=64 \
	--arch-embedding-size="3000000-3000000-3000000-3000000-3000000-3000000-3000000-3000000-3000000-3000000-3000000-3000000-3000000-3000000-3000000-3000000" \
	--arch-mlp-bot="1024-1024-1024-64" --arch-mlp-top="4096-4096-4096-4096-4096-4096-4096-1" --data-generation=random --loss-function=bce --round-targets=True \
	--learning-rate=1e-5 --mini-batch-size=512 --num-indices-per-lookup=38 --num-indices-per-lookup-fixed=True --data-size=102400 --nepochs=3 --print-freq=1 \
	--print-time --distributed --print-dist-loss  --num-batches=100000 2>&1 |tee 16_hpu_graph_seed_fix_sequential_tbl.log


