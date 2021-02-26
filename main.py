
import os
import time
import gc
import psutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL -> hides tensor warning messages

from keras.backend import clear_session
from load_data import load_data
from run_new_model import run_new_model 
from save_model import save_model

process = psutil.Process(os.getpid())

def get_memory():
    return (process.memory_info().rss - process.memory_info().shared) / 1048576 #memory usage in bytes

x_train, x_test, train_lable, test_lable = load_data()              #Load data from folder  

#for droprate in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:    
for batchsize in [64, 128, 256, 512]:   
    base_memory = get_memory()
    start_time = time.time()
    model, history = run_new_model(x_train, x_test, train_lable, test_lable , 
                                      Batchnorm = False,
                                      Droprate = 0.5, 
                                      Batchsize = batchsize, 
                                      epochs = 20)
    end_time = time.time() - start_time
    memory = get_memory() - base_memory                              #correct for data mem usage!

    print("--- %s seconds , %s mb memory in usage ---" % (end_time, memory))

    save_model(model, history, "batchsize_no_batchnorm_" + str(batchsize), end_time, memory)

    del model, history                                              #.fit() blows up ram
    gc.collect()                                                    #memory management!
    clear_session()                                                 #memory management!

