function decomp() {
    tar -xzvf "${1:-MultiTrader}".tar

}

function clean_docker() {
    docker system prune --all --volumes --force ;
}

function build_image() {
    docker build --platform=linux/amd64  -t "${1:-ethtrader1}" . ; 
    }
function test_image() {
    docker run -p 5000:5000 "${1:-ethtrader1}" ; 
    }
function tag_and_push() {
    docker tag "${1:-ethtrader1}" 905418441144.dkr.ecr.us-east-1.amazonaws.com/metalocal/${1:-ethtrader1}:latest ;
    docker push 905418441144.dkr.ecr.us-east-1.amazonaws.com/metalocal/${1:-ethtrader1}:latest;
    }
docker tag doge_trader:latest 905418441144.dkr.ecr.us-east-1.amazonaws.com/metalocal/doge_trader:latest
docker push 905418441144.dkr.ecr.us-east-1.amazonaws.com/metalocal/ethtrader1:latest


alias connect_ec2="ssh -i aws/FX_trader.pem ec2-user@ec2-34-228-167-95.compute-1.amazonaws.com
"

function clean_build_test() {
clean_docker &&\
build_image &&\
test_image &&\

}
function build_deploy() {
clean_build_test &&\
tag_and_push
}
function deploy_image() {

echo '#######################################'
echo 'COMPRESSING IMAGE'
echo '#######################################'

docker save  "${1:-ethtrader1}":bin/bashlatest | gzip >  "${1:-ethtrader1}".tar.gz 

echo '#######################################'
echo 'PUSHING IMAGE TO EC2'
echo '#######################################'

scp -i aws/FX_trader.pem  "${1:-ethtrader1}".tar.gz ec2-user@ec2-34-228-167-95.compute-1.amazonaws.com
:~/
}


function remote_run(){
ssh -i aws/FX_trader.pem ec2-user@ec2-34-228-167-95.compute-1.amazonaws.com


echo '#######################################'
echo 'DEPLOYING IMAGE'
echo '#######################################'

docker load <  "${1:-ethtrader1}".tar.gz
docker load <  jpy_trader.tar.gz

echo '#######################################'
echo 'RUNNING IMAGE'
echo '#######################################'
# sudo cat ethtrader1.bz2 | sudo docker load
sudo docker run -p 5000:5000  "${1:-ethtrader1}"
docker load <  jpy_trader.tar.gz
docker run -p 5000:5000 jpy_trader
}

# ## push image to lightsail
# aws lightsail push-container-image --service-name livetrading-service --label ethtrader1  --image ethtrader1

# ## deploy on light sail 
# aws lightsail create-container-service-deployment --service-name livetrading-service --containers file://aws/containers.json --public-endpoint file://aws/public-endpoint.json

# tar -czvf MultiTrader.tar --no-xattrs --exclude-from=ignore.txt . 

# tar -xzvf MultiTrader.tar --exclude-from=ignore.txt . 

# scp -i aws/FX_trader.pem  MultiTrader.tar ec2-user@ec2-54-242-219-93.compute-1.amazonaws.com:~/
# docker build -t fx_trader .