###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
