# Set base image (host OS)
FROM python:3.10-slim-bookworm


# COPY Pearl ./
# Install any dependencies

ADD . .

RUN apt-get install -f
RUN apt-get update
RUN apt-get install -y ca-certificates
RUN pip install --upgrade pip

RUN pip install --upgrade certifi 
    
RUN CERT_PATH=$(python -m certifi) &&\
    export SSL_CERT_FILE=${CERT_PATH}  &&\
    export REQUESTS_CA_BUNDLE=${CERT_PATH} &&\
    # export CURL_CA_BUNDLE={CERT_PATH} >> /etc/bash.bashrc
    update-ca-certificates


# RUN chmod 644 $(python -m certifi)    

# RUN CERT_PATH=$(python -m certifi) &&\
#     openssl req -new -x509 -key ${CERT_PATH} -days 3650 -out ${CERT_PATH}.crt &&\


# RUN apt-get install git -y

# RUN git config --global http.sslverify false

# RUN CERT_PATH=$(python -m certifi) &&\
#     git config --global http.sslCAinfo ${CERT_PATH} &&\
#     git config --global https.sslCAinfo ${CERT_PATH}

# RUN apt-get install -y ca-certificates§
# RUN update-ca-certificates && \
#     echo export SSL_CERT_DIR=/etc/ssl/certs >> /etc/bash.bashrc && \
#     echo export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt >> /etc/bash.bashrc
# RUN xargs -L 1 pip install < requirements.txt
# COPY certs/cacert.crt /usr/local/share/ca-certificates/cacert.crt 
        
# RUN export SSL_CERT_FILE=/usr/local/share/ca-certificates/cacert.crt
# RUN export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/cacert.crt

# RUN git clone https://github.com/sebastiancoombs/MultiTrader.git
# WORKDIR /MultiTrader/
# RUN git clone https://github.com/facebookresearch/Pearl.git


RUN pip install --upgrade pip
RUN python -m pip install p5py PEP517 wheel pyproject-toml --no-cache-dir

RUN cd Pearl && pip install -e . 


RUN xargs -L 1 pip install < requirements.txt
RUN pip install coinbase-advanced-py
RUN pip install boto3

# RUN cd ..
COPY Keys.py .
# By default, listen on port 5000
EXPOSE 5000 443 80
RUN export PYTHONWARNINGS="ignore:ResourceWarning"
RUN export PYTHONWARNINGS="ignore"

# Specify the command to run on container start
ENTRYPOINT [ "python", "-m" , "trade_coinbase" ]
# ENTRYPOINT [ "bin/bash" ]