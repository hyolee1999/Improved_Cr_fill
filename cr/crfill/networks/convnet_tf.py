import tensorflow as tf
from utils_tf import gen_conv, gen_deconv ,gen_conv1, gen_deconv1,gen_conv2, gen_deconv2, contextual_attention , resize

def contextual_module(shape,new_shape,padding='SAME',training=True):
  input = [tf.keras.Input(shape=shape, dtype='float32'),tf.keras.Input(shape=new_shape, dtype='float32'),tf.keras.Input(shape=(1), dtype='float32'),tf.keras.Input(shape=(1), dtype='float32')]
  cnum=48
  # input2 = tf.cast(input[2],tf.int32)
  # input3 = tf.cast(input[3],tf.int32)
  mask_s = resize()(input[1],input[2][0]*2,input[3][0]*2)
  x = gen_conv2(4*cnum, 5, 1, name='sconv1',training=training, padding=padding,shape_channel =None)(input[0])
  x = gen_conv2(4*cnum, 3, 1,activation ="relu", name='sconv2',training=training, padding=padding,shape_channel =None)(x)
  x , correspond = contextual_attention(x, x, mask = mask_s, ksize = 3, stride = 1, rate=2,size_1 = input[2][0],size_2 = input[3][0])
  model = tf.keras.Model(inputs = input, outputs = correspond)
  return model

def InpaintGenerator_tf_new(shape,new_shape,padding='SAME',training=True,name = "inpaintgenerator"):
    input = [tf.keras.Input(shape=shape, dtype='float32'),tf.keras.Input(shape=new_shape,dtype='float32'),tf.keras.Input(shape=new_shape,dtype='float32')]
    cnum = 48
    input0 = input[0] / 127.5  -1
    input0 = input0*(1-input[1])
    # input0 = input[0] 
    input1 = input[1] 
    input2 = input[2] 
    xin = input0
    temp = input1

    x = tf.concat([xin, input2, input2*temp], axis=3)
    x = gen_conv1(cnum, 5, 1, name='conv1',training=training, padding=padding,shape_channel =5)(x)
    x = gen_conv1(2*cnum, 3, 2, name='conv2_downsample',training=training, padding=padding,shape_channel= cnum//2)(x)
    x = gen_conv1(2*cnum, 3, 1, name='conv3',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv1(4*cnum, 3, 2, name='conv4_downsample',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv1(4*cnum, 3, 1, name='conv5',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv1(4*cnum, 3, 1, name='conv6',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv1(4*cnum, 3, rate=2, name='conv7_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv1(4*cnum, 3, rate=4, name='conv8_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv1(4*cnum, 3, rate=8, name='conv9_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv1(4*cnum, 3, rate=16, name='conv10_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv1(4*cnum, 3, 1, name='conv11',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv1(4*cnum, 3, 1, name='conv12',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_deconv1(2*cnum, name='conv13_upsample',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv1(2*cnum, 3, 1, name='conv14',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_deconv1(cnum, name='conv15_upsample',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv1(cnum//2, 3, 1, name='conv16',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv1(3, 3, 1, activation=None, name='conv17',training=training, padding=padding,shape_channel=cnum//4)(x)
    # x = tf.nn.tanh(x)
    x = tf.keras.activations.tanh(x)
    x_stage1 = x
    x = x*temp + xin*(1.-temp)
    # x_stage1 = x
    xnow = x
    x = gen_conv2(cnum, 5, 1, name='xconv1',training=training, padding=padding,shape_channel=3)(xnow)
    x = gen_conv2(cnum, 3, 2, name='xconv2_downsample',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv2(2*cnum, 3, 1, name='xconv3',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv2(2*cnum, 3, 2, name='xconv4_downsample',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv2(4*cnum, 3, 1, name='xconv5',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv2(4*cnum, 3, 1, name='xconv6',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv2(4*cnum, 3, rate=2, name='xconv7_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv2(4*cnum, 3, rate=4, name='xconv8_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv2(4*cnum, 3, rate=8, name='xconv9_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv2(4*cnum, 3, rate=16, name='xconv10_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x_hallu = x

    x = gen_conv2(cnum, 5, 1, name='pmconv1',training=training, padding=padding,shape_channel=3)(xnow)
    x = gen_conv2(cnum, 3, 2, name='pmconv2_downsample',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv2(2*cnum, 3, 1, name='pmconv3',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv2(4*cnum, 3, 2, name='pmconv4_downsample',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv2(4*cnum, 3, 1, name='pmconv5',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv2(4*cnum, 3, 1, name='pmconv6',activation="relu",training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv2(4*cnum, 3, 1, name='pmconv9',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv2(4*cnum, 3, 1, name='pmconv10',training=training, padding=padding,shape_channel=cnum*2)(x)

    pm = x
    x = tf.concat([x_hallu, pm], axis=3)
    x = gen_conv2(4*cnum, 3, 1, name='allconv11',training=training, padding=padding,shape_channel=4*cnum)(x)
    x = gen_conv2(4*cnum, 3, 1, name='allconv12',training=training, padding=padding,shape_channel=cnum*2)(x)

    x = gen_deconv2(2*cnum, name='allconv13_upsample',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv2(2*cnum, 3, 1, name='allconv14',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_deconv2(cnum, name='allconv15_upsample',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv2(cnum//2, 3, 1, name='allconv16',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv2(3, 3, 1, activation=None, name='allconv17',training=training, padding=padding,shape_channel=cnum//4)(x)

    # x = tf.nn.tanh(x)
    x = tf.keras.activations.tanh(x)
    x_stage2 = x
    
    x_stage2 = (x_stage2 + 1) * 127.5
    output = x_stage2*temp+ input[0] *(1.-temp)
    # output = (output+1.)*127.5
    # output = (output + 1) / 2

    model = tf.keras.Model(inputs = input, outputs = [output,pm],name = name)
    return model

def InpaintGenerator_tf(shape,new_shape,padding='SAME',training=True,name = "inpaintgenerator"):
    input = [tf.keras.Input(shape=shape, dtype='float32'),tf.keras.Input(shape=new_shape,dtype='float32'),tf.keras.Input(shape=new_shape,dtype='float32')]
    cnum = 48
    input0 = input[0] / 127.5  -1
    input0 = input0*(1-input[1])
    # input0 = input[0] 
    input1 = input[1] 
    input2 = input[2] 
    xin = input0
    temp = input1

    ## high res
    xin1 = tf.image.resize(xin,size = (256,256),method='nearest')
    input2 = tf.image.resize(input2,size = (256,256),method='nearest')
    mask = tf.image.resize(temp,size = (256,256),method='nearest')

    # x = tf.concat([xin, input2, input2*temp], axis=3)
    x = tf.concat([xin1, input2, input2*mask], axis=3)
    x = gen_conv(cnum, 5, 1, name='conv1',training=training, padding=padding,shape_channel =5)(x)
    x = gen_conv(2*cnum, 3, 2, name='conv2_downsample',training=training, padding=padding,shape_channel= cnum//2)(x)
    x = gen_conv(2*cnum, 3, 1, name='conv3',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv(4*cnum, 3, 2, name='conv4_downsample',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv(4*cnum, 3, 1, name='conv5',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, 1, name='conv6',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, rate=2, name='conv7_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, rate=4, name='conv8_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, rate=8, name='conv9_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, rate=16, name='conv10_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, 1, name='conv11',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, 1, name='conv12',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_deconv(2*cnum, name='conv13_upsample',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(2*cnum, 3, 1, name='conv14',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_deconv(cnum, name='conv15_upsample',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv(cnum//2, 3, 1, name='conv16',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv(3, 3, 1, activation=None, name='conv17',training=training, padding=padding,shape_channel=cnum//4)(x)
    # x = tf.nn.tanh(x)
    x = tf.keras.activations.tanh(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x_stage1 = x

    x = x_stage1*temp + xin*(1.-temp)
    # x_stage1 = x
    xnow = x
    x = gen_conv(cnum, 5, 1, name='xconv1',training=training, padding=padding,shape_channel=3)(xnow)
    x = gen_conv(cnum, 3, 2, name='xconv2_downsample',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv(2*cnum, 3, 1, name='xconv3',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv(2*cnum, 3, 2, name='xconv4_downsample',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv(4*cnum, 3, 1, name='xconv5',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv(4*cnum, 3, 1, name='xconv6',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, rate=2, name='xconv7_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, rate=4, name='xconv8_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, rate=8, name='xconv9_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, rate=16, name='xconv10_atrous',training=training, padding=padding,shape_channel=cnum*2)(x)
    x_hallu = x

    x = gen_conv(cnum, 5, 1, name='pmconv1',training=training, padding=padding,shape_channel=3)(xnow)
    x = gen_conv(cnum, 3, 2, name='pmconv2_downsample',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv(2*cnum, 3, 1, name='pmconv3',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv(4*cnum, 3, 2, name='pmconv4_downsample',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv(4*cnum, 3, 1, name='pmconv5',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, 1, name='pmconv6',activation="relu",training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, 1, name='pmconv9',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(4*cnum, 3, 1, name='pmconv10',training=training, padding=padding,shape_channel=cnum*2)(x)

    pm = x
    x = tf.concat([x_hallu, pm], axis=3)
    x = gen_conv(4*cnum, 3, 1, name='allconv11',training=training, padding=padding,shape_channel=4*cnum)(x)
    x = gen_conv(4*cnum, 3, 1, name='allconv12',training=training, padding=padding,shape_channel=cnum*2)(x)

    x = gen_deconv(2*cnum, name='allconv13_upsample',training=training, padding=padding,shape_channel=cnum*2)(x)
    x = gen_conv(2*cnum, 3, 1, name='allconv14',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_deconv(cnum, name='allconv15_upsample',training=training, padding=padding,shape_channel=cnum)(x)
    x = gen_conv(cnum//2, 3, 1, name='allconv16',training=training, padding=padding,shape_channel=cnum//2)(x)
    x = gen_conv(3, 3, 1, activation=None, name='allconv17',training=training, padding=padding,shape_channel=cnum//4)(x)

    # x = tf.nn.tanh(x)
    x = tf.keras.activations.tanh(x)
    x_stage2 = x
    
    x_stage2 = (x_stage2 + 1) * 127.5
    output = x_stage2*temp+ input[0] *(1.-temp)
    # output = (output+1.)*127.5
    # output = (output + 1) / 2

    model = tf.keras.Model(inputs = input, outputs = [output,pm],name = name)
    return model