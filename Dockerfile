# Set base image (host OS)
FROM python:3.10-slim-bookworm

COPY . .
# COPY Pearl ./
# Install any dependencies
# RUN python -m pip install p5py PEP517 wheel pyproject-toml --no-cache-dir

# ADD . .

RUN apt-get install -f
RUN apt-get update
RUN apt-get install git -y


RUN pip install --upgrade pip

# RUN xargs -L 1 pip install < requirements.txt
# RUN sudo git clone "https://github.com/facebookresearch/Pearl.git"
RUN git clone "https://github.com/sebastiancoombs/MultiTrader.git"
WORKDIR MultiTrader/

# RUN bin/bash cd MultiTrader/Pearl
RUN cd Pearl && echo ls && pip install -e . && cd .. &&  xargs -L 1 pip install < requirements.txt

# RUN cd ..

# By default, listen on port 5000
EXPOSE 5000 443

# Specify the command to run on container start
ENTRYPOINT [ "python", "-m" , "trade_coinbase" ]
# ENTRYPOINT [ "bin/bash" ]