function decomp() {
    tar -xzvf "${1:-MultiTrader}".tar

}

function clean_docker() {
    sudo docker system prune --all --volumes --force;
    }

function build_image() {
    sudo docker build -t "${1:-ethtrader1}" .; 
    }
function test_image() {
    sudo docker run -p 5000:5000 "${1:-ethtrader1}" ; 
    }

function clean_build_test() {
decomp ${2:-MultiTrader}
ls
clean_docker  &&\
build_image $1 &&\
test_image $1
}

clean_build_test fx_trader 
sudo free && sudo sync && sudo echo 3 > sudo /proc/sys/vm/drop_caches && sudo free