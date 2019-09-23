#The hyper training file to execute

"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import lenet5_cifar10
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import flag_reader
if __name__ == '__main__':
    lr_list = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3 ]
    train_batch_size_list = [16,32,64,128,256]
    #Setting the loop for setting the parameter
    for cnt,initial_lr in enumerate(lr_list):
        for cnt,train_batch_size in enumerate(train_batch_size_list):
            flags = flag_reader.read_flag()  	#setting the base case
            flags.initial_lr = initial_lr
            flags.train_batch_size = train_batch_size
            lenet5_cifar10.train_from_flags(flags)
