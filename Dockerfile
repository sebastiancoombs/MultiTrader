# Set base image (host OS)
FROM python:3.10-slim-bookworm

# COPY requirements.txt ./
# Install any dependencies
# RUN python -m pip install p5py PEP517 wheel pyproject-toml --no-cache-dir
RUN pip install --upgrade pip

ADD . .

RUN xargs -L 1 pip install < requirements.txt

RUN cd Pearl
RUN pip install -e .
RUN cd ..

# By default, listen on port 5000
EXPOSE 5000 443

# Specify the command to run on container start
ENTRYPOINT [ "python", "-m" , "trade_coinbase" ]