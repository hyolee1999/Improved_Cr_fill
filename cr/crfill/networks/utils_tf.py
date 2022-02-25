import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class gen_conv(tf.keras.layers.Layer):
    def __init__(self,cnum, ksize, stride=1, rate=1, name='conv',padding='SAME',old_padding = 'CONSTANT', activation="elu", training=True,shape_channel =None,**kwargs):
        # super(gen_conv, self).__init__()
        self.cnum = cnum
        self.ksize = ksize
        self.stride =stride
        self.rate = rate
        self.new_name = name
        self.padding = padding
        self.activation = activation
        self.training = training
        self.old_padding = old_padding
        self.shape_channel = shape_channel
        assert self.padding in ['SYMMETRIC', 'SAME', 'REFLECT']
        if self.padding == 'SYMMETRIC' or self.padding == 'REFLECT':

            self.old_padding = self.padding
            self.padding = 'VALID'
        # if shape:
        #     # pass
        #     self.layer = tf.keras.layers.Conv2D(self.cnum, self.ksize,strides = self.stride ,activation=None,padding = self.padding, dilation_rate = self.rate,name = self.new_name, input_shape = shape)
        # else:
        self.layer = tf.keras.layers.Conv2D(self.cnum, self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,name = self.new_name)
            # self.layer = tf.keras.layers.SeparableConv2D(self.cnum, self.ksize,strides = self.stride ,activation=None,padding = self.padding, dilation_rate = self.rate,name = self.new_name)
        super(gen_conv, self).__init__(**kwargs)

    # @tf.function
    def __call__(self, input):
        # if self.padding == "VALID":
        p = int(self.rate*(self.ksize-1)/2)
        input = tf.pad(tensor=input, paddings=[[0,0], [p, p], [p, p], [0,0]], mode=self.old_padding)
        x = self.layer(input)
        if self.cnum == 3 or self.activation is None:
            # conv for output
            return x
        x, y = tf.split(x, 2, 3)

        if self.activation == "elu":
            # x =  tf.nn.elu(x)
            x = tf.keras.activations.elu(x)
        elif self.activation == "relu":
            # x =  tf.nn.relu(x)
            x = tf.keras.activations.relu(x)
        # x = self.activate(x)
        # y = tf.nn.sigmoid(y)
        y = tf.keras.activations.sigmoid(y)
        z = x * y
        return z
    
    def get_config(self):

        # config = super().get_config().copy()
        config = super(gen_conv, self).get_config()
        config.update({
            'cnum': self.cnum,
            'ksize': self.ksize,
            'stride': self.stride,
            'rate': self.rate ,
            'name': self.new_name,
            'padding': self.padding,
            'activation': self.activation,
            'training': self.training,
            'old_padding':self.old_padding,
            "shape_channel":self.shape_channel
        })
        return config

@tf.keras.utils.register_keras_serializable()
class resize(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(resize, self).__init__(**kwargs)
    
    def __call__(self, input,size_1,size_2):
        x = tf.image.resize(input,size = [tf.cast(size_1[0],dtype = tf.int32),tf.cast(size_2[0],dtype = tf.int32)],method='nearest')
        # x = tf.compat.v1.image.resize_nearest_neighbor(input,size = tf.concat([size_1,size_2],axis = 0),align_corners=True)
        # x = tf.raw_ops.ResizeNearestNeighbor(images = input, size= tf.concat([size_1,size_2],axis = 0 ), align_corners=True)
        return x
    
    # @tf.function
    def get_config(self):
        config = super(resize, self).get_config()
        return config

@tf.keras.utils.register_keras_serializable()
class gen_conv1(tf.keras.layers.Layer):
    def __init__(self,cnum, ksize, stride=1, rate=1, name='conv',padding='SAME',old_padding = 'CONSTANT', activation="elu", training=True,shape_channel =None,**kwargs):
        # super(gen_conv, self).__init__()
        self.cnum = cnum
        self.ksize = ksize
        self.stride =stride
        self.rate = rate
        self.new_name = name
        self.padding = padding
        self.activation = activation
        self.training = training
        self.old_padding = old_padding
        self.shape_channel = shape_channel
        assert self.padding in ['SYMMETRIC', 'SAME', 'REFLECT']
        if self.padding == 'SYMMETRIC' or self.padding == 'REFLECT':

            self.old_padding = self.padding
            self.padding = 'VALID'
        # if shape:
        #     # pass
        # self.layer = tf.keras.layers.Conv2D(self.cnum, self.ksize,strides = self.stride ,activation=None,padding = self.padding, dilation_rate = self.rate,name = self.new_name)
        # else:
        if self.cnum == 3 or self.activation is None:
            self.layer = tf.keras.layers.Conv2D(self.cnum, self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,name = self.new_name)
            # self.layer1 = None
            # self.layer2 = None
        else:
            self.layer = tf.keras.layers.Conv2D(self.cnum//2, self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,name = self.new_name)
            self.layer1 = tf.keras.layers.Conv2D(1, self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,name = self.new_name+".mask_conv2d")
            # self.layer1 = tf.keras.layers.Conv2D(self.shape_channel,self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,groups = self.shape_channel,name = self.new_name+".mask_conv2d1")
            # self.layer2 = tf.keras.layers.Conv2D(self.cnum//2, 1,strides = 1 ,activation=None, dilation_rate = 1,name = self.new_name+".mask_conv2d2")
            # self.mask_conv2d = nn.Conv2d(in_channels=cin, out_channels=1, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)

        super(gen_conv1, self).__init__(**kwargs)

    # @tf.function
    def __call__(self, input):
        # if self.padding == "VALID":
        p = int(self.rate*(self.ksize-1)/2)
        input = tf.pad(tensor=input, paddings=[[0,0], [p, p], [p, p], [0,0]], mode=self.old_padding)
        x = self.layer(input)
        if self.cnum == 3 or self.activation is None:
            # conv for output
            return x
        # x, y = tf.split(x, 2, 3)
        y = self.layer1(input)
        # y = self.layer2(y)
        if self.activation == "elu":
            # x =  tf.nn.elu(x)
            x = tf.keras.activations.elu(x)
        elif self.activation == "relu":
            # x =  tf.nn.relu(x)
            x = tf.keras.activations.relu(x)
        # x = self.activate(x)
        # y = tf.nn.sigmoid(y)
        y = tf.keras.activations.sigmoid(y)
        z = x * y
        return z
    
    def get_config(self):

        # config = super().get_config().copy()
        config = super(gen_conv1, self).get_config()
        config.update({
            'cnum': self.cnum,
            'ksize': self.ksize,
            'stride': self.stride,
            'rate': self.rate ,
            'name': self.new_name,
            'padding': self.padding,
            'activation': self.activation,
            'training': self.training,
            'old_padding':self.old_padding,
            'shape_channel':self.shape_channel
        })
        return config

@tf.keras.utils.register_keras_serializable()
class gen_conv2(tf.keras.layers.Layer):
    def __init__(self,cnum, ksize, stride=1, rate=1, name='conv',padding='SAME',old_padding = 'CONSTANT', activation="elu", training=True,shape_channel =None,**kwargs):
        # super(gen_conv, self).__init__()
        self.cnum = cnum
        self.ksize = ksize
        self.stride =stride
        self.rate = rate
        self.new_name = name
        self.padding = padding
        self.activation = activation
        self.training = training
        self.old_padding = old_padding
        self.shape_channel = shape_channel
        assert self.padding in ['SYMMETRIC', 'SAME', 'REFLECT']
        if self.padding == 'SYMMETRIC' or self.padding == 'REFLECT':

            self.old_padding = self.padding
            self.padding = 'VALID'
        # if shape:
        #     # pass
        # self.layer = tf.keras.layers.Conv2D(self.cnum, self.ksize,strides = self.stride ,activation=None,padding = self.padding, dilation_rate = self.rate,name = self.new_name)
        # else:
        if self.cnum == 3 or self.activation is None:
            self.layer = tf.keras.layers.Conv2D(self.cnum, self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,name = self.new_name)
            # self.layer1 = None
            # self.layer2 = None
        else:
            self.layer = tf.keras.layers.Conv2D(self.cnum//2, self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,name = self.new_name)
            self.layer1 = tf.keras.layers.Conv2D(self.cnum//2, 1 ,strides = self.stride ,activation=None, dilation_rate = self.rate,name = self.new_name+".mask_conv2d")
            # self.layer1 = tf.keras.layers.Conv2D(self.shape_channel,self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,groups = self.shape_channel,name = self.new_name+".mask_conv2d1")
            # self.layer2 = tf.keras.layers.Conv2D(self.cnum//2, 1,strides = 1 ,activation=None, dilation_rate = 1,name = self.new_name+".mask_conv2d2")
            # self.mask_conv2d = nn.Conv2d(in_channels=cin, out_channels=int(cout/2), kernel_size=1, stride=stride, padding=0, dilation=rate, groups=1, bias=True)
        super(gen_conv2, self).__init__(**kwargs)

    # @tf.function
    def __call__(self, input):
        # if self.padding == "VALID":
        p = int(self.rate*(self.ksize-1)/2)
        x = tf.pad(tensor=input, paddings=[[0,0], [p, p], [p, p], [0,0]], mode=self.old_padding)
        x = self.layer(x)
        if self.cnum == 3 or self.activation is None:
            # conv for output
            return x
        # x, y = tf.split(x, 2, 3)
        y = self.layer1(input)
        # y = self.layer2(y)
        if self.activation == "elu":
            # x =  tf.nn.elu(x)
            x = tf.keras.activations.elu(x)
        elif self.activation == "relu":
            # x =  tf.nn.relu(x)
            x = tf.keras.activations.relu(x)
        # x = self.activate(x)
        # y = tf.nn.sigmoid(y)
        y = tf.keras.activations.sigmoid(y)
        z = x * y
        return z
    
    def get_config(self):

        # config = super().get_config().copy()
        config = super(gen_conv2, self).get_config()
        config.update({
            'cnum': self.cnum,
            'ksize': self.ksize,
            'stride': self.stride,
            'rate': self.rate ,
            'name': self.new_name,
            'padding': self.padding,
            'activation': self.activation,
            'training': self.training,
            'old_padding':self.old_padding,
            'shape_channel':self.shape_channel
        })
        return config

# @tf.keras.utils.register_keras_serializable()
# class gen_conv(tf.keras.layers.Layer):
#     def __init__(self,cnum, ksize, stride=1, rate=1, name='conv',padding='SAME',old_padding = 'CONSTANT', activation="elu", training=True,shape_channel =None,**kwargs):
#         # super(gen_conv, self).__init__()
#         self.cnum = cnum
#         self.ksize = ksize
#         self.stride =stride
#         self.rate = rate
#         self.new_name = name
#         self.padding = padding
#         self.activation = activation
#         self.training = training
#         self.old_padding = old_padding
#         self.shape_channel = shape_channel
#         assert self.padding in ['SYMMETRIC', 'SAME', 'REFLECT']
#         if self.padding == 'SYMMETRIC' or self.padding == 'REFLECT':

#             self.old_padding = self.padding
#             self.padding = 'VALID'
#         # if shape:
#         #     # pass
#         # self.layer = tf.keras.layers.Conv2D(self.cnum, self.ksize,strides = self.stride ,activation=None,padding = self.padding, dilation_rate = self.rate,name = self.new_name)
#         # else:
#         if self.cnum == 3 or self.activation is None:
#             self.layer = tf.keras.layers.Conv2D(self.cnum, self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,name = self.new_name)
#             # self.layer1 = None
#             # self.layer2 = None
#         else:
#             self.layer = tf.keras.layers.Conv2D(self.cnum//2, self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,name = self.new_name)
#             self.layer1 = tf.keras.layers.DepthwiseConv2D( self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,name = self.new_name+".mask_conv2d1")
#             # self.layer1 = tf.keras.layers.Conv2D(self.shape_channel,self.ksize,strides = self.stride ,activation=None, dilation_rate = self.rate,groups = self.shape_channel,name = self.new_name+".mask_conv2d1")
#             self.layer2 = tf.keras.layers.Conv2D(self.cnum//2, 1,strides = 1 ,activation=None, dilation_rate = 1,name = self.new_name+".mask_conv2d2")
#         super(gen_conv, self).__init__(**kwargs)

#     # @tf.function
#     def __call__(self, input):
#         # if self.padding == "VALID":
#         p = int(self.rate*(self.ksize-1)/2)
#         input = tf.pad(tensor=input, paddings=[[0,0], [p, p], [p, p], [0,0]], mode=self.old_padding)
#         x = self.layer(input)
#         if self.cnum == 3 or self.activation is None:
#             # conv for output
#             return x
#         # x, y = tf.split(x, 2, 3)
#         y = self.layer1(input)
#         y = self.layer2(y)
#         if self.activation == "elu":
#             # x =  tf.nn.elu(x)
#             x = tf.keras.activations.elu(x)
#         elif self.activation == "relu":
#             # x =  tf.nn.relu(x)
#             x = tf.keras.activations.relu(x)
#         # x = self.activate(x)
#         # y = tf.nn.sigmoid(y)
#         y = tf.keras.activations.sigmoid(y)
#         z = x * y
#         return z
    
#     def get_config(self):

#         # config = super().get_config().copy()
#         config = super(gen_conv, self).get_config()
#         config.update({
#             'cnum': self.cnum,
#             'ksize': self.ksize,
#             'stride': self.stride,
#             'rate': self.rate ,
#             'name': self.new_name,
#             'padding': self.padding,
#             'activation': self.activation,
#             'training': self.training,
#             'old_padding':self.old_padding,
#             'shape_channel':self.shape_channel
#         })
#         return config

@tf.keras.utils.register_keras_serializable()
class gen_deconv1(tf.keras.layers.Layer):
    def __init__(self, cnum, name='upsample', padding='SAME', training=True,shape_channel = None,**kwargs):
        # super(gen_deconv, self).__init__()
        self.cnum = cnum
        self.new_name = name
        self.padding = padding
        self.training = training
        self.shape_channel = shape_channel
        # self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.layer = gen_conv1(4*cnum, 3, 1, name=name+'_conv', padding=padding,training=training,shape_channel=shape_channel )
        super(gen_deconv1, self).__init__(**kwargs)

    # @tf.function
    def __call__(self,input):
        # x = tf.image.resize(input,size = tf.concat([size_1,size_2],axis = 0 ),method='nearest')
        # x = self.upsample(input)
        # x =  tf.keras.layers.UpSampling2D()(input)
        x = self.layer(input)
        x = tf.nn.depth_to_space(x,2)
    
        return x
    
    def get_config(self):

        # config = super().get_config().copy()
        config = super(gen_deconv1, self).get_config()
        config.update({
            'cnum': self.cnum,
            'name': self.new_name,
            'padding': self.padding,
            'training': self.training,
            'shape_channel':self.shape_channel
        })
        return config

@tf.keras.utils.register_keras_serializable()
class gen_deconv2(tf.keras.layers.Layer):
    def __init__(self, cnum, name='upsample', padding='SAME', training=True,shape_channel = None,**kwargs):
        # super(gen_deconv, self).__init__()
        self.cnum = cnum
        self.new_name = name
        self.padding = padding
        self.training = training
        self.shape_channel = shape_channel
        # self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.layer = gen_conv2(4*cnum, 3, 1, name=name+'_conv', padding=padding,training=training,shape_channel=shape_channel )
        super(gen_deconv2, self).__init__(**kwargs)

    # @tf.function
    def __call__(self,input):
        # x = tf.image.resize(input,size = tf.concat([size_1,size_2],axis = 0 ),method='nearest')
        # x = self.upsample(input)
        # x =  tf.keras.layers.UpSampling2D()(input)
        x = self.layer(input)
        x = tf.nn.depth_to_space(x,2)
    
        return x
    
    def get_config(self):

        # config = super().get_config().copy()
        config = super(gen_deconv2, self).get_config()
        config.update({
            'cnum': self.cnum,
            'name': self.new_name,
            'padding': self.padding,
            'training': self.training,
            'shape_channel':self.shape_channel
        })
        return config


@tf.keras.utils.register_keras_serializable()
class gen_deconv(tf.keras.layers.Layer):
    def __init__(self, cnum, name='upsample', padding='SAME', training=True,shape_channel = None,**kwargs):
        # super(gen_deconv, self).__init__()
        self.cnum = cnum
        self.new_name = name
        self.padding = padding
        self.training = training
        self.shape_channel = shape_channel
        # self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.layer = gen_conv(cnum, 3, 1, name=name+'_conv', padding=padding,training=training,shape_channel=shape_channel )
        super(gen_deconv, self).__init__(**kwargs)

    # @tf.function
    def __call__(self,input):
        # x = tf.image.resize(input,size = tf.concat([size_1,size_2],axis = 0 ),method='nearest')
        # x = self.upsample(input)
        x =  tf.keras.layers.UpSampling2D()(input)
        x = self.layer(x)
        # x = tf.nn.depth_to_space(x,2)
    
        return x
    
    def get_config(self):

        # config = super().get_config().copy()
        config = super(gen_deconv, self).get_config()
        config.update({
            'cnum': self.cnum,
            'name': self.new_name,
            'padding': self.padding,
            'training': self.training,
            'shape_channel':self.shape_channel
        })
        return config


def contextual_attention(f, b,size_1,size_2,batch=1, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True,dynamic = True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    # raw_fs = tf.shape(input=f)
    raw_fs = tf.concat([tf.constant([batch]),tf.cast(size_1*2,dtype = tf.int32),tf.cast(size_2*2,dtype = tf.int32),tf.constant([96])],axis = 0)
    if dynamic:
        # raw_int_fs = tf.shape(f)
        raw_int_bs = tf.shape(b)
        # temp = f.get_shape().as_list()
        # new_int_bs = [None]*4
        # new_int_bs[0] = tf.cast(raw_int_bs[0], tf.int32)
        # new_int_bs[1] = tf.cast(raw_int_bs[1], tf.int32)
        # new_int_bs[2] = tf.cast(raw_int_bs[2], tf.int32)
        # new_int_bs[3] = tf.cast(raw_int_bs[3], tf.int32)
        # new_int_bs_0 = tf.cast(raw_int_bs[0], tf.int32)
        # new_int_bs_1 = tf.cast(raw_int_bs[1], tf.int32)
        # new_int_bs_2 = tf.cast(raw_int_bs[2], tf.int32)
        # new_int_bs_3 = tf.cast(raw_int_bs[3], tf.int32)
    else:
        raw_int_fs = f.get_shape().as_list()
        raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.image.extract_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    if dynamic:
        # raw_w = tf.reshape(raw_w, [new_int_bs[0], -1, kernel, kernel, new_int_bs[3]])
        # raw_w = tf.reshape(raw_w, [new_int_bs_0, -1, kernel, kernel, new_int_bs_3])
        raw_w = tf.reshape(raw_w, [batch, -1, kernel, kernel, 96])
    else:
        raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(a=raw_w, perm=[0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    # ///////////////////////
    # f = resize(f, scale=1./rate, func=tf.image.resize)
    # f = tf.image.resize(f[0],size = tf.concat([size_1[0],size_2[0]],axis = 0 ),method='nearest')
    f = resize()(f,size_1,size_2)
    #////////////////////////
    if dynamic:
        # b = resize(b, to_shape=[tf.cast(new_int_bs[1]/rate,tf.int32), tf.cast(new_int_bs[2]/rate,tf.int32)], func=tf.image.resize,dynamic = False)
        # b = resize(b[:], to_shape=tf.concat([size_1/rate,size_2/rate],axis = 0 ), func=tf.image.resize,dynamic = False)
        # b = tf.image.resize(b[:,:,:,:],size = tf.concat([size_1[0],size_2[0]],axis = 0 ),method='nearest')
        b = resize()(b,size_1,size_2)
    else:
        b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        # mask = resize(mask, scale=1./rate, func=tf.image.resize)
        # mask = resize(mask, to_shape=tf.concat([size_1,size_2],axis = 0 ), func=tf.image.resize)
        mask = resize()(mask, size_1,size_2)
    # fs = tf.shape(input=f)
    fs = tf.concat([tf.constant([batch]),tf.cast(size_1,dtype = tf.int32),tf.cast(size_2,dtype = tf.int32),tf.constant([96])],axis = 0 )
    if dynamic:
        int_fs = tf.shape(input=f)
        # new_int_fs = [None] * 4
        # new_int_fs[0] = tf.cast(int_fs[0],tf.int32)
        # new_int_fs[1] = tf.cast(int_fs[1],tf.int32)
        # new_int_fs[2] = tf.cast(int_fs[2],tf.int32)
        # new_int_fs[3] = tf.cast(int_fs[3],tf.int32)
        # new_int_fs_0 = tf.cast(int_fs[0],tf.int32)
        # new_int_fs_1 = tf.cast(int_fs[1],tf.int32)
        # new_int_fs_2 = tf.cast(int_fs[2],tf.int32)
        # new_int_fs_3 = tf.cast(int_fs[3],tf.int32)
    else:
        int_fs = f.get_shape().as_list()

    if dynamic:
        #fix
        # f_groups = f
        f_groups = tf.split(f, batch, axis=0)
    else:
        f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    # bs = tf.shape(input=b)
    bs = tf.concat([tf.constant([batch]),tf.cast(size_1,dtype = tf.int32),tf.cast(size_2,dtype = tf.int32),tf.constant([96])],axis = 0 )
    if dynamic:
        int_bs = tf.shape(input=b)
        new_int_bs = [None] * 4
        new_int_bs[0] = tf.cast(int_bs[0],tf.int32)
        new_int_bs[1] = tf.cast(int_bs[1],tf.int32)
        new_int_bs[2] = tf.cast(int_bs[2],tf.int32)
        new_int_bs[3] = tf.cast(int_bs[3],tf.int32)
    else:
        int_bs = b.get_shape().as_list()
    # w = tf.image.extract_patches(
    #     b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.image.extract_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    if dynamic:
        # w = tf.reshape(w, [new_int_fs[0], -1, ksize, ksize, new_int_fs[3]])
        # w = tf.reshape(w, [new_int_fs_0, -1, ksize, ksize, new_int_fs_3])
        #fix
        # w = tf.reshape(w, [1, -1, ksize, ksize, 96])
        w = tf.reshape(w, [batch, -1, ksize, ksize, 96])
    else:
        w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(a=w, perm=[0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.image.extract_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(a=m, perm=[0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(input_tensor=m, axis=[0,1,2], keepdims=True), 0.), tf.float32)
    if dynamic:
        # w_groups = w
        # raw_w_groups = raw_w
        # fix
        w_groups = tf.split(w, batch, axis=0)
        raw_w_groups = tf.split(raw_w, batch, axis=0)
    else:
        w_groups = tf.split(w, int_bs[0], axis=0)
        raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    correspond = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    #fix
    if  dynamic:
        xi = f_groups[0]
        wi = w_groups[0]
        raw_wi = raw_w_groups[0]
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(input_tensor=tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(input=xi, filters=wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(input=yi, filters=fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(a=yi, perm=[0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(input=yi, filters=fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(a=yi, perm=[0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask
        corr = yi
        correspond.append(corr)
        # offset = tf.argmax(input=yi, axis=3, output_type=tf.int32)
        # offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        # print(yi.shape)
        # print(wi_center.shape)
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        # offsets.append(offset)
    else:
        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            # conv for compare
            wi = wi[0]
            wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(input_tensor=tf.square(wi), axis=[0,1,2])), 1e-4)
            yi = tf.nn.conv2d(input=xi, filters=wi_normed, strides=[1,1,1,1], padding="SAME")

            # conv implementation for fuse scores to encourage large patches
            if fuse:
                yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
                yi = tf.nn.conv2d(input=yi, filters=fuse_weight, strides=[1,1,1,1], padding='SAME')
                yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
                yi = tf.transpose(a=yi, perm=[0, 2, 1, 4, 3])
                yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
                yi = tf.nn.conv2d(input=yi, filters=fuse_weight, strides=[1,1,1,1], padding='SAME')
                yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
                yi = tf.transpose(a=yi, perm=[0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

            # softmax to match
            yi *=  mm  # mask
            yi = tf.nn.softmax(yi*scale, 3)
            yi *=  mm  # mask

            offset = tf.argmax(input=yi, axis=3, output_type=tf.int32)
            offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
            # deconv for patch pasting
            # 3.1 paste center
            wi_center = raw_wi[0]
            yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
            y.append(yi)
            offsets.append(offset)
    y = tf.concat(y, axis=0)
    correspond = tf.concat(correspond,axis = 0)
    # if dynamic:
    #     y.set_shape(temp)
    # else:
    #     y.set_shape(raw_int_fs)
    # offsets = tf.concat(offsets, axis=0)
    # if dynamic:
    #     temp = b.get_shape().as_list()
    #     offsets.set_shape(temp[:3] + [2])
    # else:
    #     offsets.set_shape(int_bs[:3] + [2])
    # # case1: visualize optical flow: minus current position
    # h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    # w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    # offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    # flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    # if rate != 1:
    #     flow = resize(flow, scale=rate, func=tf.compat.v1.image.resize_bilinear)
    # return y, flow
    return y , correspond

