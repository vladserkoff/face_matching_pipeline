FROM debian:8

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -qq update && \
    apt-get -qq -y install \
        bzip2 \
        ca-certificates \
        curl \
        dpkg \
        git \
        grep \
        libgl1-mesa-glx \
        locales \
        nano \
        sed \
        wget \
    && apt-get -qq -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    SHELL=/bin/bash

RUN TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` \
    && curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb \
    && dpkg -i tini.deb \
    && rm tini.deb \
    && apt-get clean
ENTRYPOINT [ "/usr/bin/tini", "--" ]

RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda config --append channels conda-forge \
    && conda clean --all --yes

ENV NAME facematch

COPY environment.yml /tmp
RUN conda env create -f /tmp/environment.yml && \
    conda clean --all --yes
ENV PATH /usr/local/envs/$NAME/bin:$PATH

WORKDIR /opt/$NAME

ADD $NAME ./$NAME
ADD api.py .

EXPOSE 8000

CMD gunicorn --bind=0.0.0.0:8000 api:app
