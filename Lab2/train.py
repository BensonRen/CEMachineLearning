#The hyper training file to execute

"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import lenet5_cifar10
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import flag_reader
if __name__ == '__main__':
    lr_list = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3 ]
    #Setting the loop for setting the parameter
    for i in range(5):
        flags = flag_reader.read_flag()  	#setting the base case
        flags.lr = lr_list[i]
        lenet5_cifar10.train_from_flag(flags)
