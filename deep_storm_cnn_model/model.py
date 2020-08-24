from tensorflow.keras.layers import Input,UpSampling2D,MaxPooling2D,Convolution2D,BatchNormalization,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from data_generator.gaussian_kernel_2d import gaussian_kernel_2d
from tensorflow.keras import losses
import tensorflow as tf

psf_heatmap = gaussian_kernel_2d(7,1)
gfilter = tf.reshape(psf_heatmap, [7, 7, 1, 1])

def L1L2loss(input_shape):
    def bump_mse(heatmap_true, spikes_pred):

        # generate the heatmap corresponding to the predicted spikes
        heatmap_pred = K.conv2d(spikes_pred, gfilter, strides=(1, 1), padding='same')

        # heatmaps MSE
        loss_heatmaps = losses.mean_squared_error(heatmap_true,heatmap_pred)

        # l1 on the predicted spikes
        loss_spikes = losses.mean_absolute_error(spikes_pred,tf.zeros(input_shape))
        return loss_heatmaps + loss_spikes
    return bump_mse

def conv_bn_relu(nb_filter, rk, ck, name):
    def f(input):
        conv = Convolution2D(nb_filter, kernel_size=(rk, ck), strides=(1,1),
                               padding="same", use_bias=False,
                               kernel_initializer="Orthogonal",name='conv-'+name)(input)

        conv_norm = BatchNormalization(name='BN-'+name)(conv)

        conv_norm_relu = Activation(activation = "relu",name='Relu-'+name)(conv_norm)

        return conv_norm_relu

    return f
def CNN(input,names):
    Features1 = conv_bn_relu(32,3,3,names+'F1')(input)
    pool1 = MaxPooling2D(pool_size=(2,2),name=names+'Pool1')(Features1)
    Features2 = conv_bn_relu(64,3,3,names+'F2')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2),name=names+'Pool2')(Features2)
    Features3 = conv_bn_relu(128,3,3,names+'F3')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2),name=names+'Pool3')(Features3)
    Features4 = conv_bn_relu(512,3,3,names+'F4')(pool3)
    up5 = UpSampling2D(size=(2, 2),name=names+'Upsample1')(Features4)
    Features5 = conv_bn_relu(128,3,3,names+'F5')(up5)
    up6 = UpSampling2D(size=(2, 2),name=names+'Upsample2')(Features5)
    Features6 = conv_bn_relu(64,3,3,names+'F6')(up6)
    up7 = UpSampling2D(size=(2, 2),name=names+'Upsample3')(Features6)
    Features7 = conv_bn_relu(32,3,3,names+'F7')(up7)
    return Features7

def buildModel(input_dim):
    input_ = Input (shape = (input_dim))
    act_ = CNN (input_,'CNN')
    density_pred = Convolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same",\
                                  activation="linear", use_bias = False,\
                                  kernel_initializer="Orthogonal",name='Prediction')(act_)
    model = Model (inputs= input_, outputs=density_pred)
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss = L1L2loss(input_dim))
    return model

