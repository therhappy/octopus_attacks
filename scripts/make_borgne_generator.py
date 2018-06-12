import matplotlib
matplotlib.use('Agg')
import numpy as np 
import pandas as pd
import keras.backend as K
print('Standard imports OK')

# Import custom functions
from make_borgne_generator_functions import *
print('Custom imports OK')

# Load parameters
exec(open('make_borgne_generator_parameters.py','r').read())
print('Parameters OK \n Starting training...')

# Start training
K.clear_session()
logger = borgne_train(img_shape, noise_shape, nb_class, target_class, target_path, image_path, save_model_path, nb_epochs, batch_size, log_freq, print_freq, save_freq)
    
pd.DataFrame(logger).to_csv('logs_borgne_training.csv')