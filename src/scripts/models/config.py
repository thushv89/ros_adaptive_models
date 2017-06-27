CONTEXT_WINDOW_SIZE = 1

#INPUT SIZE (128,96)
TF_SCOPES = ['conv1','conv2','fc1']
TF_VAR_SHAPES = [[5,5,3,32],[3,3,32,64],[49152,128]]
TF_STRIDES = {'conv1':[1,2,2,1],'conv2':[1,2,2,1]}
TF_DECONV_STR = 'deconv'
TF_WEIGHTS_STR = 'weights'
TF_BIAS_STR = 'bias'