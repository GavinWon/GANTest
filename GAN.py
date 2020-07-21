# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:22:41 2020

@author: Gavin
"""

#The Discriminator
from keras.layers import Conv2D, BatchNormalization, Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU

# function for building the discriminator layers
def build_discriminator(start_filters, spatial_dim, filter_size):
    
    # function for building a CNN block for downsampling the image
    def add_discriminator_block(x, filters, filter_size):
      x = Conv2D(filters, filter_size, padding='same')(x)
      x = BatchNormalization()(x)
      x = Conv2D(filters, filter_size, padding='same', strides=2)(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(0.3)(x)
      return x
    
    # input is an image with shape spatial_dim x spatial_dim and 3 channels
    inp = Input(shape=(spatial_dim, spatial_dim, 3))

    # design the discrimitor to downsample the image 4x
    x = add_discriminator_block(inp, start_filters, filter_size)
    x = add_discriminator_block(x, start_filters * 2, filter_size)
    x = add_discriminator_block(x, start_filters * 4, filter_size)
    x = add_discriminator_block(x, start_filters * 8, filter_size)
    
    # average and return a binary output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inp, outputs=x)


##The Generator
from keras.layers import Deconvolution2D, Reshape

def build_generator(start_filters, filter_size, latent_dim):
  
  # function for building a CNN block for upsampling the image
  def add_generator_block(x, filters, filter_size):
      x = Deconvolution2D(filters, filter_size, strides=2, padding='same')(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(0.3)(x)
      return x

  # input is a noise vector 
  inp = Input(shape=(latent_dim,))

  # projection of the noise vector into a tensor with 
  # same shape as last conv layer in discriminator
  x = Dense(4 * 4 * (start_filters * 8), input_dim=latent_dim)(inp)
  x = BatchNormalization()(x)
  x = Reshape(target_shape=(4, 4, start_filters * 8))(x)

  # design the generator to upsample the image 4x
  x = add_generator_block(x, start_filters * 4, filter_size)
  x = add_generator_block(x, start_filters * 2, filter_size)
  x = add_generator_block(x, start_filters, filter_size)
  x = add_generator_block(x, start_filters, filter_size)    

  # turn the output into a 3D tensor, an image with 3 channels 
  x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)
  
  return Model(inputs=inp, outputs=x)