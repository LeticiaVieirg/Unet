from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTrasnpose, Concatenate, Input
from tensorflow.keras.models import Model

# a convolution block w/ two conv 3x3
#input recurse

def conv_block (inputs, num_filters): 
    x=Conv2D(num_filters, 3, padding="same")(inputs) #receive num filters, kernel 3x3 and same padding
    x=BatchNormalization()(x) #input ocult
    x=Activation("relu")(x)

    x=Conv2D(num_filters, 3, padding="same")(x) 
    x=BatchNormalization()(x) #input ocult
    x=Activation("relu")(x)

    return x


def enconder_block(inputs, num_filters):
    x=conv_block(inputs, num_filters)
    p=MaxPool2D((2,2))(x)
    return x,p


#step 2x2 doble the recurse resolution and same 
def decoder_block(inputs, skip_feartures, num_filters):
    x=Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(inputs)
    x=Concatenate()([x, skip_feartures])
    x=conv_block(x, num_filters)

    return x


def build_unet(input_shape):
    inputs=Input(input_shape)

    # encoder
    s1, p1=enconder_block(inputs, 64)
    s2, p2=enconder_block(p1, 128)
    s3, p3=enconder_block(p2, 256)
    s4, p4=enconder_block(p3, 512)

    #bridge
    b1=conv_block(p4, 1024)

    #decoder
    #star the double resolution
    d1=decoder_block(b1, s4, 512)
    d2=decoder_block(d1, s3, 256)
    d3=decoder_block(d2, s2, 128)
    d4=decoder_block(d3, s1, 64)

    #output
    # some 1 layer (camada)
    outputs=Conv2D(1, (1,1), padding="same", activation="sigmoid")(d4)

    model=Model(inputs, outputs, name="U-Net")
    return model

if __name__=="__main__":
    input_shape=(512,512,3)
    model=build_unet(input_shape)
    model.summary()
