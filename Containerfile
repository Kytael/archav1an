FROM cachyos:latest

RUN pacman-key --init && \
    pacman-key --populate archlinux cachyos

WORKDIR /scripts
