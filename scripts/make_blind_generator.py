'''
This file is intended to train one or several generators on an array of similar models in order to extract
training data.
'''

# Start
print('Starting model blind inversion protocol ...')

# General parameters
target_model_path = '../target_models/'
save_model_path = '../attackers/blind_inversion/'
save_images_path = '../images/blind_inversion/'
log_path = '../logs/blind_inversion/'
models_to_train = 2
# Training parameters
# CURRENT: white-box MNIST
target_class = 2
nb_class = 10
epochs = 500
target_class = 0

# Library imports
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import os
import numpy as np
print('Library imports OK')
from make_blind_generator_functions import build_generator, train_adversarial
print('Local imports OK')

# Import target models
print('Importing target models from {} ...'.format(target_model_path))
list_dir_target = os.listdir(target_model_path)
list_target_models = []
for elm in list_dir_target:
    if elm[-3:] == '.h5': list_target_models.append(elm)
del list_dir_target
nb_targets = len(list_target_models)
print('{} target models found'.format(nb_targets))

for i,target_model in enumerate(list_target_models):
    print('Attacking model {} out of {}'.format(i+1,nb_targets))
    target = load_model(target_model_path + target_model)
    for n in range(models_to_train):
        print('|- Generating attacker {} out of {}'.format(n+1, models_to_train))
        opt = Adam()
        generator = build_generator()
        train_adversarial(
              generator,
              target,
              opt,
              target_class,
              nb_class,
              epochs,
              white_box=True,
              batch_size=128,
              sample_interval=100,
              sample_path = save_images_path,
              sample_name = 'model_{}_{}'.format(i,n)
              )
        generator.save(save_model_path + 'model_{}_{}.h5'.format(i,n))
        print('|- Model saved')

print('All attackers generated and trained')
