###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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
#
# Custom manylinux_2_26 for building pytorch-integration.
#
# Only supports Python 3.8 - unlike typical manylinux images.
#
# It's based on AL2, our oldest supported OS glibc-wise,
# so we can build using it and run everywhere.
#
# As we're dropping AL2 in the near future, this is a minimum-effort image.
# When we switch to manylinux_2_28, a lighter-weight solution should be crafted.
#

# Why try harder?
FROM artifactory-kfs.habana-labs.com/devops-docker-local/habana-builder-pt:amzn2-1.17.0

# Disabling mirror from base image as it's already dead
RUN yum-config-manager --disable mirror.centos.org_centos_7_sclo_x86_64_rh_ && \
    yum install -y passwd && \
    yum clean all && \
    rm -rf /var/cache/yum

COPY entrypoint /usr/local/bin/
ENTRYPOINT [ "entrypoint" ]
CMD [ "/bin/bash" ]
