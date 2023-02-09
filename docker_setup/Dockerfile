FROM ubuntu:latest

ARG USER=ebana-user

RUN apt-get update && apt-get -y install sudo \
    && apt-get install -y git wget vim ngspice libc6-dev\
    && wget http://ftp.fr.debian.org/debian/pool/main/n/ngspice/libngspice0_30.2-1~bpo9+1_amd64.deb && apt install -y ./libngspice0_30.2-1~bpo9+1_amd64.deb \
    && wget http://ftp.fr.debian.org/debian/pool/main/n/ngspice/libngspice0-dev_30.2-1~bpo9+1_amd64.deb && apt install -y ./libngspice0-dev_30.2-1~bpo9+1_amd64.deb \
    && apt-get install -y python3 python3-pip \
    && pip3 install pyspice scikit-learn

RUN adduser --disabled-password \--gecos '' $USER \
    && adduser $USER sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER $USER

RUN git clone https://github.com/mawatfa/ebana /home/$USER/ebana

WORKDIR /home/$USER/ebana
