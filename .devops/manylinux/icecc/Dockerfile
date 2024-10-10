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

FROM artifactory-kfs.habana-labs.com/docker/manylinux/al2

COPY setup-icecc.sh /setup-icecc.sh
RUN bash /setup-icecc.sh && rm /setup-icecc.sh
ENV PATH=/opt/icecream/bin:/opt/icecream/sbin:$PATH

EXPOSE 10246 8765/TCP 8765/UDP 8766
COPY icecc-entrypoint /usr/local/bin
ENTRYPOINT [ "icecc-entrypoint" ]
CMD [ "/bin/bash" ]
