import numpy as np
from PIL import Image
import sys
sys.path.append('../python/')
import caffe
import scipy.misc as misc
from collections import OrderedDict
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe


def printParams(net):
	net = net.params
	keyslist = [key for key in net.keys()]
	tmp = OrderedDict()
	for (i, name) in enumerate(keyslist):
		tmp[name] = net[name][0].data.shape
		print name, tmp[name]


def printBlobs(net):
	net = net.blobs
	keyslist = [key for key in net.keys()]

	for (i, name) in enumerate(keyslist):
		if 'split' in name: continue
		if len(net[name].data.shape) == 0:
			print name, net[name].data
		else:
			print name, net[name].data[0].shape


def compareArchs(net1, net2, type='params'):
	keyslist1 = [key for key in net1.keys()]
	keyslist2 = [key for key in net2.keys()]
	# print '\t', n1[:-9],'\t',n2[:-9]
	for n1,n2 in zip(keyslist1,keyslist2):
		if type=="params": # show params
			print n1,": ", net1[n1][0].data.shape, '\t|',n2, net2[n2][0].data.shape
		else: # show data
			print n1,": ", net1[n1].data.shape, '\t|',n2, net2[n2].data.shape

def predict(im, net, meanval):

	#assert(type(meanval)==types.TupleType)
	# im = im.resize((500,500))
	in_ = np.array(im, dtype=np.float32)
	in_ = in_[:,:,::-1]
	in_ -= meanval
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


def montage(data, start_dim=0, dim_num=1, tile_num = 100):
    channels, height, width = data.shape
    if start_dim + dim_num > channels:
        print 'start_dim exceeds the range'
#         break
    if height*tile_num > 5000:
        tile_num = np.floor(5000/height)
        print 'reduce tile_num to ', tile_num, ' for memory consideration'

    imw = int(np.floor(np.sqrt(tile_num)))
    img = np.zeros((dim_num, height*imw, width*imw))

    for i in range(imw):
        for j in range(imw):
            tmp = data[start_dim:start_dim+dim_num,:,:]
            if (tmp.max()-tmp.min()) > 0:
                tmp = (tmp-tmp.min()) / (tmp.max()-tmp.min())
            img[:, i*height:(i+1)*height, j*width:(j+1)*width] = tmp
            start_dim += 1
            if start_dim >= channels: 
                img = img.transpose((1,2,0))
                img = np.squeeze(img)
                return img
    img = img.transpose((1,2,0))
    img = np.squeeze(img)
    return img
	
