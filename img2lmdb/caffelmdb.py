# convert images in files to lmdb and mostly concern convert structured labels rather than standard (key, value) format. In fact, it is achieved by create to sepearate lmdb for data and label. 
# this is the examples for fully convolution network
# CODE ZZZ
import lmdb
import scipy.misc as misc
import scipy
import os
import glob
import sys
sys.path.append('/home/zizhaozhang/caffe/github/caffe/python')
import caffe
from random import shuffle
import numpy as np


class img2lmdb:

	def __init__(self,savepath):
		self.savepath = savepath

	def read(self, name='fcn-train-lmdb', n=0):
		# read nth image in dbfile
		name = os.path.join(self.savepath,name)
		env = lmdb.open(name,readonly=True)
		f =  env.begin()
		cursor = f.cursor()
		keylist = [key for (key, value) in cursor]

		raw_d = f.get(keylist[n])

		datum = caffe.proto.caffe_pb2.Datum()
		datum.ParseFromString(raw_d)

		x = scipy.fromstring(datum.data,dtype=scipy.uint8)
		x = x.reshape(datum.channels, datum.height, datum.width)
		y = datum.label
		print 'label:' 
		print "shape: ", x.shape
		if x.shape[0] == 3:
			x = x.transpose((1,2,0))
			x = x[:,:,::-1] # convert to RGB
		else:
			x = x[0]
		return x
		

	def write(self, files, imgpath, typ, name='fcn-train-lmdb', sufix='.jpg'):
		name = os.path.join(self.savepath,name)
		dbfile = lmdb.open(name,map_size=int(1e12))
			
		meanrgb = np.zeros(3) # for mean RGB
		mag = 1. * len(files)
				
		imgnames = [os.path.split(name)[1] for name in files]
		print imgnames
		
		with dbfile.begin(write=True) as txn:
			for (i, img) in enumerate(imgnames):
				I = misc.imread(os.path.join(imgpath,img[:-4]+sufix))
				#misc.imshow(I)
				if len(I.shape) == 3: 
					#meanrgb += np.sum(np.sum(img,0),0) / mag
					I = I[:,:,::-1] # change to BGR
					I = scipy.transpose(I,(2,0,1)) # change to CxHxW
				else:# white and black, normally it is 
					I = I.reshape((I.shape[0],I.shape[1],1))
					I = I.transpose((2,0,1))
				if I.dtype != typ:
					print "convert to ", typ
					I = I.astype(typ)
				print i, ':', img,'|', I.dtype,'|',  I.shape,'|',  'min: ', I.min(),'|',  'max: ', I.max()
				form = '{:0>10d}'.format(i)
				try:
					datum = caffe.io.array_to_datum(I)
					txn.put(form, datum.SerializeToString())
				except:
					print "write " + name + "error"
				
		dbfile.close()
		meanrgb = meanrgb*mag/len(files)
		print "mean value R, G, B: ", meanrgb


	def checkgt(files, imgpath, name='fcn-train-lmdb', sufix='.jpg'):
		name = os.path.join(self.savepath,name)
		min = 1e5
		max = -1

		imgnames = [os.path.split(name)[1] for name in files]
		for (i, img) in enumerate(imgnames):
			I = misc.imread(os.path.join(imgpath,img[:-4]+sufix))
			if len(I.shape) == 3: 
				I = I[:,:,::-1] # change to BGR
				I = scipy.transpose(I,(2,0,1)) # change to CxHxW
			else:#white and black
				I = I.reshape((I.shape[0],I.shape[1],1))
				I = I.transpose((2,0,1))
			print i, ':', img, type(I), I.shape
			print 'min{}, max{}'.format(I.min(),I.max())
			if I.min() < min:
				min = I.min()
			if I.max() > max:
				max = I.max()
		
		print "mask value range from {} to {}".format(min, max)

if __name__ == '__main__':
	path = '/home/zizhaozhang/caffe/github/caffe/FCN/59_context_labels/'
	searchbase = os.path.join(path,'*.png')
	imgpath = '/home/zizhaozhang/caffe/github/caffe/FCN/JPEGImages'
	files = glob.glob(searchbase)
	shuffle(files)
	hd = img2lmdb('/home/zizhaozhang/caffe/github/caffe/FCN/')
	#hd.write(files[:4000],imgpath=imgpath)
	#hd.write(files[:4000],name='../fcn-train-label-lmdb',sufix='.png')
	#hd.write(files[4000:],name='../fcn-test-lmdb',imgpath=imgpath)
	#hd.write(files[4000:],name='../fcn-test-label-lmdb',sufix='.png')

	
