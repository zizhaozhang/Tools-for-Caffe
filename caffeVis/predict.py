try:
	import caffe
except:
	print "can not find caffe"

import types

def predict(im, net, meanval):

	assert(type(meanval)==types.TupleType)
	# im = im.resize((500,500))
	in_ = np.array(im, dtype=np.float32)
	in_ = in_[:,:,::-1]
	in_ -= np.array(meanval)
	in_ = in_.transpose((2,0,1))

	keyslist = net.blobs.keys()
	datatype = keyslist[0]
	outtype = keyslist[-1]
	# shape for input (data blob is N x C x H x W), set data
	net.blobs[datatype].reshape(1, *in_.shape)
	net.blobs[datatype].data[...] = in_
	# run net and take argmax for prediction
	score = net.forward()
	out = net.blobs[outtype].data[0].argmax(axis=0)
	return out, net
