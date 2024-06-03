# Set base image (host OS)
FROM python:3.10-bullseye

# RUN conda --version
COPY . .

# By default, listen on port 5000
EXPOSE 5000/tcp

# Install any dependencies

RUN pip install p5py PEP517 wheel

RUN python -m pip install pyproject-toml

RUN python -m pip install -r requirements.txt 


# Specify the command to run on container start
CMD [ "python", "-m" , "trade_app" ]