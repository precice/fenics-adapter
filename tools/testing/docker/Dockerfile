# Dockerfile to build a ubuntu image containing the installed Debian package of a release 
FROM ubuntu:24.04

# Add the precice user to run test with mpi
RUN useradd -m -s /bin/bash precice
ENV PRECICE_USER=precice

# Installing necessary dependencies
RUN apt-get -qq update && \
    apt-get -qq install software-properties-common && \
    add-apt-repository -y ppa:fenics-packages/fenics && \
    apt-get -qq update && \
    apt-get -qq install --no-install-recommends fenics && \
    rm -rf /var/lib/apt/lists/*