# Set base image (host OS)
FROM python:3.10-slim-bookworm


# COPY Pearl ./
# Install any dependencies


RUN apt-get install -f
RUN apt-get update
RUN apt-get install -y ca-certificates
## update the certificates
RUN pip install --upgrade pip
RUN pip install --upgrade certifi 

## try to ignore warnings but it doesnt seet to work
RUN export PYTHONWARNINGS="ignore:ResourceWarning"
RUN export PYTHONWARNINGS="ignore"

RUN CERT_PATH=$(python -m certifi) &&\
    export SSL_CERT_FILE=${CERT_PATH}  &&\
    export REQUESTS_CA_BUNDLE=${CERT_PATH} &&\
    # export CURL_CA_BUNDLE={CERT_PATH} &&\
    update-ca-certificates


## install git so we can clone the repo
# RUN apt-get install git -y
# RUN git config --global http.sslverify false

## Clone the repository
# RUN git clone https://github.com/sebastiancoombs/MultiTrader.git
## change work dir to the cloned folder
# WORKDIR MultiTrader/
## Clone the neural network lib
# RUN git clone https://github.com/sebastiancoombs/Pearl

# instead of installing from Git install form local
ADD . .

## install the python libraries
RUN python -m  pip install p5py PEP517 wheel pyproject-toml --no-cache-dir

## install pearl stuff
RUN cd Pearl && pip install -e . 

## install multi trader stuff
RUN xargs -L 1 python -m pip install < requirements.txt

## copy the keys for coinbase and stuff
COPY Keys.py .

# By default, listen on port 5000,443,80
EXPOSE 5000 443 80

# Specify the command to run on container start
ENTRYPOINT [ "python", "-m" , "trade_coinbase" ]
