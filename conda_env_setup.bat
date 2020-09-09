setlocal
set ENV_NAME="NeuralNetworkFromScratch"
call conda create -n %ENV_NAME% python=3.8 -y
call conda activate %ENV_NAME%
call pip install tensorflow~=2.3.0
call pip install matplotlib~=3.3.1

endlocal