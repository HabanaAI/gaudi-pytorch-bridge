#!/bin/bash

CONT=$1
VERSION=$2
DOCK_NUM=$3
LINUX_VER=$4

if [ -z "$4" ]
then
LINUX_VER=20
else
LINUX_VER=$4
fi

if [ -z "$5" ]
then
PT_VER=1.10.1
else
PT_VER=$5
fi

if [[ -z "$3" ]]; then
echo "HOSTS=\"node1 node2\" docker_multi.sh  <container name> <release version number> <build number> <optional, default 20:Linux version number>"
exit
fi

var=${HOSTS:-localhost}
for HOSTNAME in ${var} ; do
    M_HLS+=${HOSTNAME}
    M_HLS+=","
done

M_HLS=${M_HLS::-1}


USERNAME=labuser
PASSWD=Hab12345
SCRIPT="$PYTORCH_MODULES_ROOT_PATH/topologies/tools/docker_single.sh ${CONT} ${VERSION} ${DOCK_NUM} ${LINUX_VER} ${PT_VER} ${M_HLS}"
echo $SCRIPT
count=0
for HOSTNAME in ${var} ; do
    echo "Connecting to $HOSTNAME"
    #echo "sshpass -p ${PASSWD} ssh ${USERNAME}@${HOSTNAME} ${SCRIPT}"
    sshpass -p ${PASSWD} ssh ${USERNAME}@${HOSTNAME} ${SCRIPT}
    echo "Finished setting up $HOSTNAME"
    count=$((count+1))
done

DOCK=${VERSION}-${DOCK_NUM}

CONT_NAME=${CONT}-${DOCK}
if [ "$count" -gt "1" ]; then
FIRSTHOST=`echo $HOSTS | head -n1 | awk '{print $1;}'`
sshpass -p ${PASSWD} ssh ${USERNAME}@${FIRSTHOST} docker exec --privileged ${CONT_NAME} python -u $PYTORCH_MODULES_ROOT_PATH/topologies/tools/setup_multihls.py
fi
echo "Connect to one of the first host and login to the container using the below command :"
echo "docker exec --privileged -it ${CONT_NAME} bash"