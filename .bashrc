function clean_docker() {

    docker system prune --all --volumes --force;}

function build_image() {
    echo '#######################################'
    echo "Building image with name: ${1:-ethtrader1}"
    echo '#######################################'
    docker build --platform=linux/amd64  -t "${1:-ethtrader1}" .
    }
function test_image() {
    echo '#######################################'
    echo "Testing Image ${1:-ethtrader1}"
    echo '#######################################'
    docker run -p 5000:5000 "${1:-ethtrader1}" 
 }
alias connect_ec2="ssh -i aws/ethtrader.pem ec2-user@ec2-54-242-219-93.compute-1.amazonaws.com"

function tag_and_push() {
    echo '#######################################'
    echo "Tagging and Pushing Image with name ${1:-ethtrader1}"
    echo '#######################################'
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 905418441144.dkr.ecr.us-east-1.amazonaws.com
    docker tag "${1:-ethtrader1}" 905418441144.dkr.ecr.us-east-1.amazonaws.com/${1:-ethtrader1}:latest ;
    docker push 905418441144.dkr.ecr.us-east-1.amazonaws.com/${1:-ethtrader1}:latest;
    }

function build_test() {

build_image ${1:-ethtrader1}&&\
test_image ${1:-ethtrader1}
}

function clean_build_test() {
clean_docker ${1:-ethtrader1}&&\
build_test ${1:-ethtrader1}
}
function build_deploy() {
build_test ${1:-ethtrader1}&&\
tag_and_push ${1:-ethtrader1}
}

function clean_build_deploy() {
clean_build_test ${1:-ethtrader1}&&\
tag_and_push ${1:-ethtrader1}
}
# ## push image to lightsail
# aws lightsail push-container-image --service-name livetrading-service --label ethtrader1  --image ethtrader1

# ## deploy on light sail 
# aws lightsail create-container-service-deployment --service-name livetrading-service --containers file://aws/containers.json --public-endpoint file://aws/public-endpoint.json

# tar -czvf MultiTrader.tar --no-xattrs --exclude-from=ignore.txt . 

# tar -xzvf MultiTrader.tar --exclude-from=ignore.txt . 

# scp -i aws/FX_trader.pem  MultiTrader.tar ec2-user@ec2-54-242-219-93.compute-1.amazonaws.com:~/
# docker build -t fx_trader .