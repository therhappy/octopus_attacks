from keras.layers import Input, Dense, Activation, Conv2D, UpSampling2D, Reshape, Flatten, Dropout, BatchNormalization
from keras.engine import Model
from keras.models import load_model
from keras.optimizers import Adam
import keras.backend as K 
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def custom_loss_0(y_true, y_pred, w_catcross=1, w_wass=0.01):
    cat_cross = K.categorical_crossentropy(y_true, y_pred)
    wass = wasserstein_loss(y_true,y_pred)
    return w_catcross * cat_cross + w_wass * wass

class ShadowLoss():
    '''
    intended to imitate BEGAN loss in BORGNE shadow setting
    (where autoencoder architecture is irrelevant ...)
    crafted from issue#4813 and
    github.com/mokemokechicken/keras_BEGAN/blob/master/src/began/training.py
    # Arguments
        k_init: Float; initial k factor
        lambda_k: Float; k learning rate
        gamma: Float; equilibrium factor
    '''
    
    __name__ = 'shadow_loss'
    
    def __init__(self, initial_k=0.001, lambda_k=0.001, gamma=0.5):
        self.lambda_k = lambda_k
        self.gamma = gamma
        self.k_var = K.variable(initial_k, dtype=K.floatx(), name="shadow_k")
        self.m_global_var = K.variable(np.array([0]), dtype=K.floatx(), name="m_global")
        self.updates=[]

    def __call__(self, y_true, y_pred):  # y_true, y_pred shape: (batch_size, nb_class)
        # LET'S MAKE A STRONG HYPOTHESIS: BATCH IS HALF NOISE & HALF GENERATED
        # ORDERED AS  EVEN NUMBERS = NOISE & ODD NUMBERS = GENERATED
        noise_true, generator_true = y_true[:, ::2], y_true[:, 1::2] #even, odd
        noise_pred, generator_pred = y_pred[:, ::2], y_pred[:, 1::2] #even, odd
        loss_noise = K.mean(K.abs(noise_true - noise_pred))
        loss_generator = K.mean(K.abs(generator_true - generator_pred))
        began_loss = loss_noise - self.k_var * loss_generator
        
        # The code from which this is adapted used an update mechanism
        # where DiscriminatorModel collected Loss Function's updates attributes
        # This is replaced here by LossUpdaterModel (hihihi)

        mean_loss_noise = K.mean(loss_noise)
        mean_loss_gen = K.mean(loss_generator)
        
        # update K
        new_k = self.k_var + self.lambda_k * (self.gamma * mean_loss_noise - mean_loss_gen)
        new_k = K.clip(new_k, 0, 1)
        self.updates.append(K.update(self.k_var, new_k))

        # calculate M-Global
        m_global = mean_loss_noise + K.abs(self.gamma * mean_loss_noise - mean_loss_gen)
        m_global = K.reshape(m_global, (1,))
        self.updates.append(K.update(self.m_global_var, m_global))

        return began_loss
    
    @property
    def k(self):
        return K.get_value(self.k_var)

    @property
    def m_global(self):
        return K.get_value(self.m_global_var)



class LossUpdaterModel(Model):
    """Model which collects updates from loss_func.updates"""

    @property
    def updates(self):
        updates = super().updates
        if hasattr(self, 'loss_functions'):
            for loss_func in self.loss_functions:
                if hasattr(loss_func, 'updates'):
                    updates += loss_func.updates
        return updates



def build_classifier(img_shape, nb_cat):
    inputs = Input(shape=img_shape)
    foo = Conv2D(128, (3, 3), padding='valid', data_format='channels_last')(inputs)
    foo = Activation('relu')(foo)
    foo = Dropout(0.25)(foo)
    foo = Flatten()(foo)
    foo = Dense(128)(foo)
    foo = Activation('relu')(foo)
    foo = Dropout(0.25)(foo)
    outputs = Dense(nb_cat, activation='sigmoid')(foo)
    model = LossUpdaterModel(inputs, outputs)
    return model



def build_generator(img_shape=(28,28,1), noise_shape=(1,)):
    '''Construct generator for borgne setting'''

    # Pictures dimensions
    IMG_HEIGHT, IMG_WIDTH, NB_CHANNELS_INPUTS = img_shape

    # Input layer
    inputs = Input(shape=noise_shape)
    foo = Dense(7*7, activation='linear')(inputs)
    foo = BatchNormalization()(foo)

    img = Reshape((7,7,1))(foo) # tensor (batch, x, y, channel)
    foo = Conv2D(128, kernel_size=(3,3), padding='same')(img)
    foo = Activation('relu')(foo)
    # for MNIST : ?,7,7,128

    foo = UpSampling2D(size=(2, 2), data_format='channels_last')(foo)
    foo = Conv2D(filters=128, kernel_size=(3,3), padding="same")(foo)
    foo = Activation('relu')(foo)
    # for MNIST : ?,14,14,64

    foo = UpSampling2D(size=(2, 2), data_format='channels_last')(foo)
    foo = Conv2D(filters=64, kernel_size=(3,3), padding="same")(foo)
    foo = Activation('relu')(foo)
    # for MNIST : ?,28,28,32

    foo = Conv2D(filters=1, kernel_size=(1,1), padding="same")(foo)
    # for MNIST : ?,28,28,1
    
    #A sigmoid layer to have outputs between 0 and 1
    predictions = Activation('sigmoid')(foo)

    model = Model(inputs=inputs, outputs=predictions)
    return model

def build_generator(img_shape, noise_shape):
    '''simplistic, for tests only'''
    inputs = Input(shape=noise_shape)
    foo = Dense(np.prod(img_shape), activation='relu') (inputs)
    outputs = Reshape(img_shape) (foo)
    model = Model(inputs=inputs, outputs=outputs)
    return model



def sample_images(epoch, generator, noise_size, image_path):
    '''Call during training to print samples of images generated'''
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, noise_size))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(image_path + '%10.3d.png' % epoch)
    plt.close()

def borgne_train(img_shape, noise_shape, nb_class, target_class, target_path, image_path, save_model_path, nb_epochs, batch_size=32, log_freq=100, print_freq=1000, save_freq=5000):
        # Initialize optimizers
    g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Initialize base entities
    shadow = build_classifier(img_shape, nb_class)
    generator = build_generator(img_shape=img_shape, noise_shape=noise_shape)
    target = load_model(target_path)
    print('Models retrieved')

    # Build graphs
    noise_size = noise_shape[0]
    z = Input(shape=noise_shape)
    img = generator(z)
    out_class = shadow(img)
    combined = Model(z, out_class)
    print('Graphs OK')

    # Compile models
    shadow.trainable = True
    shadloss = ShadowLoss()
    shadow.compile(loss=shadloss,
                   optimizer= g_opt,
                   metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    shadow.trainable = False
    combined.compile(loss=custom_loss_0,
                     optimizer=d_on_g_opt,
                     metrics=['categorical_accuracy'])
    shadow.trainable = True
    print('Compilation OK')

    # Prepare logs
    logger = []
    
    # Prepare to split the batch
    mid = int(batch_size/2)

    # Training loop ########################################"
    for epoch in range(nb_epochs):
        
        # set seed for replication
        np.random.seed(epoch)

        # train shadow
        noise = np.random.normal(0,1,(mid, noise_size))
        shadow_X1 = generator.predict(noise) # make half the batch a prediction
        shadow_X2 = np.random.normal(0,1,(batch_size - mid,) + img_shape) # other half is noise
        shadow_y1 = target.predict(shadow_X1)
        shadow_y2 = target.predict(shadow_X2)
        shadow_X, shadow_y = np.zeros((batch_size,) + img_shape), np.zeros((batch_size, nb_class))
        # ShadowLoss currently required even=noise,  odd=generated
        shadow_X[::2], shadow_y[::2] = shadow_X2, shadow_y2
        shadow_X[1::2], shadow_y[1::2] = shadow_X1, shadow_y1
        s_loss, s_cat, s_tc = shadow.train_on_batch(shadow_X, shadow_y)

        # train generator
        #noise = np.random.normal(0,1,(batch_size, noise_size))
        noise = np.ones((batch_size, noise_size))
        combined_y = np.zeros((batch_size, nb_class))
        combined_y[:,target_class] = 1
        g_loss, g_cat = combined.train_on_batch(noise, combined_y)
        
        # print & log module
        if epoch % print_freq == 0:
            print(epoch, s_loss, g_loss, sep=' -/- ')
            sample_images(epoch, generator, noise_size, image_path)
            logger.append([epoch, s_loss, g_loss, s_cat, s_tc, g_cat])
        elif epoch % log_freq == 0:
            logger.append([epoch, s_loss, g_loss, s_cat, s_tc, g_cat])
        if epoch !=0 and epoch % save_freq == 0:
            shadow.save(save_model_path + 'current_shadow.h5')
            generator.save(save_model_path + 'current_generator.h5')
            combined.save(save_model_path + 'current_combined.h5')
            print('Models saved')
    
    shadow.save(save_model_path + 'trained_shadow.h5')
    generator.save(save_model_path + 'trained_generator.h5')
    combined.save(save_model_path + 'trained_combined.h5')

    logger.append([nb_epochs, s_loss, g_loss, s_cat, s_tc, g_cat])
    sample_images(nb_epochs, generator, noise_size, image_path)
    print('Training is over')
    
    return logger