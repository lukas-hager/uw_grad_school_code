cd /Users/hlukas/git/uw_grad_school_code/CSE-546/hw3

conda init bash
conda activate cse446

python3 homeworks/intro_pytorch/crossentropy_search.py > /Users/hlukas/git/uw_grad_school_code/CSE-546/hw3/logs/ce_search_log.txt
python3 homeworks/intro_pytorch/mean_squared_error_search.py > /Users/hlukas/git/uw_grad_school_code/CSE-546/hw3/logs/mse_search_log.txt
python3 homeworks/neural_network_mnist/main.py > /Users/hlukas/git/uw_grad_school_code/CSE-546/hw3/logs/mnist_log.txt