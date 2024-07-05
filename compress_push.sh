

function compress_dir() {
    tar -czvf "${1:-MultiTrader}".tar --no-xattrs --exclude-from=ignore.txt . ; 
    }

function push_dir() {
    scp -i aws/ethtrader.pem  "${1:-MultiTrader}".tar ec2-user@ec2-3-90-235-151.compute-1.amazonaws.com:~/ ; 
    }

function connect_ec2() {
    ssh -i aws/ethtrader.pem ec2-user@ec2-3-90-235-151.compute-1.amazonaws.com ; 
    }

function press_push() {
compress_dir $1 &&\
push_dir $1 &&\
connect_ec2
}
rm MultiTrader.tar
press_push MultiTrader
