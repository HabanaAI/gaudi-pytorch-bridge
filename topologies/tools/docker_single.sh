#!/bin/bash
#Update the mount points accordingly to point to the model_garden
MOUNT="/software/lfs/data/:/root/data -v  /software:/software "
#Set the PATH_SRC to point to qnpu repo (habanaqa, model_garden and automationto) use the QA scripts
PATH_SRC=""


#PROXY_SETTING="-e HTTP_PROXY=http://proxy-dmz.intel.com:911 -e https_proxy=http://proxy-dmz.intel.com:912 -e http_proxy=http://proxy-dmz.intel.com:911 \
#-e no_proxy=habana-labs.com,127.0.0.1,localhost -e NO_PROXY=habana-labs.com,127.0.0.1,localhost -e HTTPS_PROXY=http://proxy-dmz.intel.com:912"


if [ "$PATH_SRC" == "" ];
then
echo "Set the PATH_SRC to point to the src repo path. Need to point to the root dir of model_garden"
exit
fi

if [[ -z "$3" ]]; then
echo "setup_docker.sh  <container name> <release version number> <build number> <optional, default 18:Linux version number>"
exit
fi

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
PT_VER="1.11.0"
else
PT_VER=$5
fi


HOSTNAMES=$6

entrypoint()
{
cat <<EOF>/tmp/entry_start.sh
#!/bin/bash
ln -s $PATH_SRC/habanaqa/ /root/habanaqa
ln -s $PATH_SRC/model_garden/ /root/model_garden
ln -s $PATH_SRC/automation/ /root/automation
sed -i 's/#Port 22/Port 3022/g' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
echo 'TCPKeepAlive yes' >> /etc/ssh/sshd_config
echo 'ClientAliveInterval 20' >> /etc/ssh/sshd_config
echo 'ClientAliveCountMax 120' >> /etc/ssh/sshd_config
echo root:4Hbqa??? | chpasswd
mkdir -p ~/.ssh
echo -e "Host *\n  ServerAliveInterval 20\n  ServerAliveCountMax 120" >> ~/.ssh/config
service ssh restart
echo -en "[global]\nindex-url = https://artifactory.habana-labs.com/api/pypi/pypi-virtual/simple" > /etc/pip.conf
exec bash
EOF
chmod +x /tmp/entry_start.sh
}

entrypoint
echo "$CONT $VERSION $DOCK_NUM $LINUX_VER"
UBNT="ubuntu$LINUX_VER.04"

#if [ "$VERSION"  -gt "1" ]; then
#    VERSION="0.${VERSION}.0"
#else
#    VERSION="${VERSION}.0.0"
#fi

DOCK=${VERSION}-${DOCK_NUM}
echo "Docker version : $DOCK"
#echo "docker images | grep ${UBNT} | grep ${DOCK} | tr -s ' ' | cut -d' ' -f 3"
IMG_ID=`docker images | grep ${UBNT} | grep ${DOCK} | tr -s ' ' | cut -d' ' -f 3`
echo $IMG_ID
if [ -z "$IMG_ID" ]
then
echo "Image was not found, fetching"
echo "docker pull artifactory-kfs.habana-labs.com/docker-local/${VERSION}/${UBNT}/habanalabs/pytorch-installer-$PT_VER:${DOCK}"
RES=$(docker pull artifactory-kfs.habana-labs.com/docker-local/${VERSION}/${UBNT}/habanalabs/pytorch-installer-$PT_VER:${DOCK} 2>&1 | grep "Error")
echo $RES
fi

if [ -z "$RES" ]
then
IMG_ID=`docker images | grep ${UBNT} | grep ${DOCK} | tr -s ' ' | cut -d' ' -f 3`
echo "Found Image $IMG_ID"
CONT_NAME=${CONT}-${DOCK}
CHECK_CONT=`docker ps  | grep ${CONT_NAME} | awk -F' '  '{print $NF}'`

    if [ ! -z "$HOSTNAMES" ]
    then
     if [ "$CHECK_CONT" ==  "$CONT_NAME" ]; then
        echo "Found container $CONT_NAME, removing it"
        docker stop $CONT_NAME
        docker rm $CONT_NAME
        CHECK_CONT=""
     fi
    fi
    if [ "$CHECK_CONT" ==  "$CONT_NAME" ]; then
        echo "Found container $CONT_NAME"
        else
        echo "Creating container $CONT_NAME"
        docker run -d -t --privileged=true --env BUILD_NUMBER=$DOCK_NUM  ${PROXY_SETTING} \
        --env MULTI_HLS_IPS=$HOSTNAMES \
        --env SOFTWARE_DATA=/software/data/ \
        --env SOFTWARE_LFS_DATA=/software/lfs/data/ \
        --env MODEL_GARDEN_PYTORCH_PATH=/root/model_garden/PyTorch/ \
        --env HABANA_SOFTWARE_STACK=/root \
        -e DISPLAY= -v  $MOUNT -v /tmp:/tmp  \
        --entrypoint /tmp/entry_start.sh  \
        --net=host --ipc=host  --user root --name ${CONT_NAME} --workdir=/root  $IMG_ID  bash
        echo "Success"
    fi
    if [ -z "${HOSTNAMES}" ]
    then
            docker exec --privileged -it ${CONT_NAME} bash
            echo "Do you want to stop the container and remove it y/n"
            read userinput
            if [ "$userinput" = "y" ]; then
                docker stop ${CONT_NAME}
                docker rm ${CONT_NAME}
            fi
    else
        docker exec --privileged  ${CONT_NAME} pip install pexpect
        docker exec --privileged  ${CONT_NAME} pip install -r automation/ci/requirements-test.txt
    fi

else
echo "failure"
fi


