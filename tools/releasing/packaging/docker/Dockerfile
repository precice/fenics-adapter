# Dockerfile to build a ubuntu image containing the installed Debian package of a release 
ARG PYTHON_BINDINGS_REF=develop
ARG from=precice/python-bindings:${PYTHON_BINDINGS_REF}
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
ARG FENICS_ADAPTER_REF=develop

# Building fenics-adapter
RUN python3 -m pip install --user git+https://github.com/precice/fenics-adapter.git@$FENICS_ADAPTER_REF
