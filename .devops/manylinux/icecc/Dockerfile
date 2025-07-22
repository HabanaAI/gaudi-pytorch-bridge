###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

FROM artifactory-kfs.habana-labs.com/docker/manylinux/manylinux_2_28:latest

COPY setup-icecc.sh /setup-icecc.sh
RUN bash /setup-icecc.sh && rm /setup-icecc.sh && dnf clean all
ENV PATH=/opt/icecream/bin:/opt/icecream/sbin:$PATH

# Work around IceCC's bug with packaging compiler aliases
RUN rm /opt/rh/gcc-toolset-13/root/usr/bin/{cc,c++}

EXPOSE 10246 8765/TCP 8765/UDP 8766
COPY icecc-entrypoint.sh /usr/local/bin
ENTRYPOINT [ "icecc-entrypoint.sh" ]
