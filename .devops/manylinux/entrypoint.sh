#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

if [ -z "$HOST_USER" ] || [ -z "$HOST_UID" ] || [ -z "$HOST_GID" ]; then
    # shellcheck disable=SC2016
    echo 'Please add the following to your docker run invocation: -e HOST_USER=$USER -e HOST_UID=$(id -u) -e HOST_GID=$(id -g)'
    echo "as it's needed to avoid file permission issues after building"
    exit 1
fi

chmod ugo+rwX /tmp

groupadd -g "$HOST_GID" software
useradd -m -u "$HOST_UID" -g "$HOST_GID" "$HOST_USER" 2>/dev/null
passwd -d "$HOST_USER" >/dev/null 2>&1
USER_HOME=/home/"$HOST_USER"

echo "$HOST_USER ALL=(ALL) ALL" >> /etc/sudoers

chown "$HOST_UID":"$HOST_GID" "$USER_HOME"
if [[ ! -d /.ssh-host ]]; then
    echo "Please add the following to your docker run invocation: -v ~/.ssh:/.ssh-host:ro"
    echo "to mount you host user's SSH directory (read-only)"
    exit 1
fi

cp -a /.ssh-host "$USER_HOME"/.ssh
chown -R "$HOST_USER" "$USER_HOME"/.ssh
chmod go-rwx "$USER_HOME"/.ssh

cat >> "$USER_HOME"/.ssh/config << EOM
Host *
    User $HOST_USER
    StrictHostKeyChecking no
EOM

cat >> "$USER_HOME"/.bashrc << EOM
[ -z "$PS1" ] && export PS1='[\u@\h \W]\\$ '
EOM

exec su "$HOST_USER" -c "cd-entrypoint.sh $*"
