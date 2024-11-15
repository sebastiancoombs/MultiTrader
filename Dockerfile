# Set base image (host OS)
FROM python:3.10-slim-bookworm


# COPY Pearl ./
# Install any dependencies

# ADD . .

RUN apt-get install -f
RUN apt-get update
RUN apt-get install git -y
RUN git config --global http.sslverify false


RUN openssl req -newkey rsa:4096 \
-x509 \
-sha256 \
-days 3650 \
-nodes \
-out cert.crt \
-keyout cert.key \
# -passout pass:password 2048 \
-subj "/C=US/ST=California/L=Sandiego/O=Metalocal/OU=traderdept/CN=seb"\
    &&\
cp cert.crt /usr/local/share/ca-certificates/cert.crt \
    &&\
update-ca-certificates
RUN export SSL_CERT_FILE=/usr/local/share/ca-certificates/cert.crt
RUN export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/cert.crt
# RUN apt-get install -y ca-certificates§
# RUN update-ca-certificates && \
#     echo export SSL_CERT_DIR=/etc/ssl/certs >> /etc/bash.bashrc && \
#     echo export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt >> /etc/bash.bashrc
# RUN xargs -L 1 pip install < requirements.txt
RUN git clone https://github.com/sebastiancoombs/MultiTrader.git

WORKDIR /MultiTrader/

RUN git clone https://github.com/facebookresearch/Pearl.git
RUN ls -la

RUN pip install --upgrade pip
RUN python -m pip install p5py PEP517 wheel pyproject-toml --no-cache-dir

RUN cd Pearl && pip install -e . 
# WORKDIR /MultiTrader/

RUN xargs -L 1 pip install < requirements.txt
RUN pip install coinbase-advanced-py
RUN pip install boto3
RUN pip install --upgrade certifi
# RUN cd ..
COPY Keys.py .
# By default, listen on port 5000
EXPOSE 5000 443 80
RUN export PYTHONWARNINGS="ignore:ResourceWarning"
RUN export PYTHONWARNINGS="ignore"

# Specify the command to run on container start
ENTRYPOINT [ "python", "-m" , "trade_coinbase" ]
# ENTRYPOINT [ "bin/bash" ]