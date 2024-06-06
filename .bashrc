function clean_docker() {docker system prune --all --volumes --force;}

function build_image() {docker build --platform=linux/amd64  -t "${1:-ethtrader1}" .; }
function test_image() {docker run -p 5000:5000 "${1:-ethtrader1}" ; }
alias connect_ec2="ssh -i aws/ethtrader.pem ec2-user@ec2-54-242-219-93.compute-1.amazonaws.com"

function clean_build_test() {
clean_docker &&\
build_image &&\
test_image
}

function deploy_image() {

echo '#######################################'
echo 'COMPRESSING IMAGE'
echo '#######################################'

docker save  "${1:-ethtrader1}":latest | gzip >  "${1:-ethtrader1}".tar.gz 

echo '#######################################'
echo 'PUSHING IMAGE TO EC2'
echo '#######################################'

scp -i aws/ethtrader.pem  "${1:-ethtrader1}".tar.gz ec2-user@ec2-54-242-219-93.compute-1.amazonaws.com:~/
}


function remote_run(){
ssh -i aws/ethtrader.pem ec2-user@ec2-54-242-219-93.compute-1.amazonaws.com

echo '#######################################'
echo 'DEPLOYING IMAGE'
echo '#######################################'

docker load <  "${1:-ethtrader1}".tar.gz

echo '#######################################'
echo 'RUNNING IMAGE'
echo '#######################################'
# sudo cat ethtrader1.bz2 | sudo docker load
sudo docker run -p 5000:5000  "${1:-ethtrader1}"
}

## push image to lightsail
aws lightsail push-container-image --service-name livetrading-service --label ethtrader1  --image ethtrader1

## deploy on light sail 
aws lightsail create-container-service-deployment --service-name livetrading-service --containers file://aws/containers.json --public-endpoint file://aws/public-endpoint.json