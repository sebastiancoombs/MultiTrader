# Set base image (host OS)
FROM python:3.10-bookworm
RUN apt-get install -y ca-certificates

# RUN conda --version
COPY . .

# By default, listen on port 5000
EXPOSE 5000/tcp
EXPOSE 443/tcp
EXPOSE 443

# Install any dependencies

RUN pip install p5py PEP517 wheel

RUN python -m pip install pyproject-toml

RUN python -m pip install -r requirements.txt 


# Specify the command to run on container start
ENTRYPOINT [ "python", "-m" , "trade_app" ]