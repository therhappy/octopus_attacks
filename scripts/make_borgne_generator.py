import matplotlib
import datetime
matplotlib.use('Agg')
import numpy as np 
import pandas as pd
import keras.backend as K

# Import custom functions
from make_borgne_generator_functions import *

try:
    # Load parameters
    exec(open('make_borgne_generator_parameters.py','r').read())
    
    print('Parameters OK \n Starting training...')
    # Start training
    K.clear_session()
    logger, state = borgne_train(img_shape, noise_shape, nb_class, target_class, target_path, image_path, save_model_path, nb_epochs, observe, batch_size, log_freq, print_freq, save_freq)
    logfile = logfile + '_' + datetime.datetime.now().strftime("%m-%d-%I%M%p")
    if state == 0:
        print('Training finished: saving logs to {}'.format(logfile))
        pd.DataFrame(logger).to_csv(logfile)
    else:
        print('Training interrupted: saving logs to {}'.format(logfile))
        logfile = logfile + '_interrupted'
        pd.DataFrame(logger).to_csv(logfile)

except KeyboardInterrupt:
    logfile = logfile + '_' + datetime.datetime.now().strftime("%m-%d-%I%M%p") + '_interrupted'
    print("Shutdown request received. Dumping to {}".format(logfile))
    pd.DataFrame(logger).to_csv(logfile)