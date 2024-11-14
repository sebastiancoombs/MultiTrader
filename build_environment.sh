function install_pearl() {
    # git clone https://github.com/facebookresearch/Pearl.git;
    cd Pearl;
    pip install -e .;
    cd ..;

}

function ordered_pip_install() {
    xargs -L 1 pip install < requirements.txt;
    }

function make_env() {
    conda deactivate;
    conda create -n pearl_env python=3.10.13 -y;
    conda activate pearl_env; 
    conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
    conda deactivate;
    conda activate pearl_env;
    }



function build() {
make_env && install_pearl && ordered_pip_install 

}

build
