# Dockerfile to build a ubuntu image containing the installed Debian package of a release 
ARG branch=develop
ARG from=precice/python-bindings:${branch}
FROM $from

USER root
# Installing necessary dependencies
RUN add-apt-repository -y ppa:fenics-packages/fenics && \
    apt-get -qq update && \
    apt-get -qq install --no-install-recommends fenics && \
    rm -rf /var/lib/apt/lists/*

USER precice

# Upgrade pip to newest version (pip version from 18.04 apt-get is outdated)
RUN python3 -m pip install --user --upgrade pip

# Rebuild image if force_rebuild after that command
ARG CACHEBUST
ARG branch=develop

# Building fenics-adapter
RUN python3 -m pip install --user numpy==1.14.5 https://github.com/precice/fenics-adapter/archive/$branch.zip
