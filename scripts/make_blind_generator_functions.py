from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

def build_generator(img_shape=(28,28,1), noise_shape=(1,)):
    # Pictures dimensions
    IMG_HEIGHT, IMG_WIDTH, NB_CHANNELS_INPUTS = img_shape
    
    # Input layer
    inputs = Input(shape=noise_shape)
    foo = Dense(np.prod(img_shape), activation='softplus')(inputs)
    img = Reshape(img_shape)(foo)
    
    #Now, let us build the decoder
    conv7 = Conv2D(filters=128, kernel_size=(3,3), padding="same")(img)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('softplus')(conv7)

    conv8 = Conv2D(filters=64, kernel_size=(3,3), padding="same")(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('softplus')(conv8)

    #conv8 = Cropping2D(cropping=((0,5),(0,5)))(conv8)

    #Add the last layer to have an output with the input height and width but only two channels
    conv9 = Conv2D(filters=1, kernel_size=(1,1), padding="same")(conv8)
    #A sigmoid layer to have outputs between 0 and 1
    predictions = Activation('sigmoid')(conv9)
    #Finally, let us build the model
    model = Model(inputs=inputs, outputs=predictions)

    return model


def sample_images(generator, epoch, path='images/', name='extract', rows=5, columns=5):
    r, c = rows, columns
    #noise = np.random.normal(0, 1, (r * c, 100))
    #noise[0,:] = 1
    noise = np.ones((r*c,1))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    if r*c != 1:
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
    else:
        fig = plt.figure()
        plt.imshow(gen_imgs[0, :,:,0], cmap='gray')
    if epoch < 100 : fig.savefig(path+name+'0{}.png'.format(epoch))
    else: fig.savefig(path+name+'{}.png'.format(epoch))
    plt.close()

def train_adversarial(
                    generator,
                    classifier,
                    opt,
                    target_class,
                    nb_class,
                    epochs,
                    white_box=True,
                    batch_size=128,
                    sample_interval=50,
                    sample_path = 'images/',
                    sample_name = 'extract'
                    ):
    '''
    This is adversarial setting : much resembles a GAN but only train the generator.
    Theorically we only have black-box access to the discriminator. This might be dealt with
    shadow copies of the attacked model.
    '''
    # The generator wants to build samples that maximize certainty of target class
    # build target array
    target_y = np.zeros((batch_size, nb_class))
    target_y[:,target_class] = 1
    print('target is', target_y[0])
    
    if white_box:
        # Assuming we have white box access, for instance to a shadow copy of the model
        # For the combined model we will only train the generator
        classifier.trainable = False
        # The discriminator takes generated images as input and determines validity
        z = Input(shape=(1,))
        img = generator(z)
        valid = classifier(img)
        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        combined = Model(z, valid)
        opt = Adam(lr=0.001)
        combined.compile(loss='binary_crossentropy',
                         optimizer=opt,
                         metrics=['mean_absolute_error'])
    else:
        print('ERROR :: White-box access is currently required')
        raise NotImplementedError()

    for epoch in range(epochs):
        # ---------------------
        #  Train Generator
        # ---------------------
        #noise = np.random.normal(0, 1, (batch_size, 100))
        noise = np.ones((batch_size,1))
        g_loss = combined.train_on_batch(noise, target_y)

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0: sample_images(generator, epoch,
                                                        path = sample_path,
                                                        columns=1, rows=1,
                                                        name = sample_name)