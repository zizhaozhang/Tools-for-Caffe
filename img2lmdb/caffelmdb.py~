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
import matplotlib.pyplot as plt

class img2lmdb:

	def __init__(self,savepath):
		self.savepath = savepath

	def read(self, typ, name='fcn-train-lmdb', n=0):
		# read nth image in dbfile
		#print name, typ
		name = os.path.join(self.savepath,name)
		env = lmdb.open(name,readonly=True)
		f =  env.begin()
		cursor = f.cursor()
		keylist = [key for (key, value) in cursor]

		raw_d = f.get(keylist[n])

		datum = caffe.proto.caffe_pb2.Datum()
		datum.ParseFromString(raw_d)
		
		#x = scipy.fromstring(datum.data,dtype=typ) # deprecated when dtype is float need to change to datum.float_data
		x = caffe.io.datum_to_array(datum)
		#print datum.channels, datum.height, datum.width, x.nbytes
		x = x.reshape(datum.channels, datum.height, datum.width)
		y = datum.label
		#print 'label:', y
		#print "shape: ", x.shape
		if x.shape[0] == 3:
			x = x.transpose((1,2,0))
			x = x[:,:,::-1] # convert to RGB
		else:
			x = x[0]
			print type(x)
			#print "seg label: ", x.min() , ','. x.max()
		return x

	def checklmdb(self, name1, typ1, name2, typ2, num, path):
		for i in range(num):

			print 'wirte out #{} image'.format(i)
			img = self.read(typ1,name1,i)
			gt = self.read(typ2,name2,i)
			gt = gt.astype(img.dtype)
			
			#gt = gt.reshape((gt.shape[0],gt.shape[1],1)).repeat(3,axis=2)
			#gt = np.reshape(gt,(gt.shape[0],gt.shape[1],1))
			#img = np.reshape(img[:,:,0],(gt.shape[0],gt.shape[1]))
			print "seg label: ", gt.max(),gt.min()
			#com = np.concatenate((img,gt),axis=1)
			#misc.imsave(os.path.join(path, str(i)+'.png'),com)
			plt.figure(1)
			plt.subplot(121)
			plt.imshow(img)
			plt.subplot(122)
			plt.imshow(gt,cmap='Accent')
			plt.savefig(os.path.join(path, str(i)+'.png'))

	def write(self, files, imgpath, typ, name='fcn-train-lmdb', sufix='.jpg'):
		name = os.path.join(self.savepath,name)
		dbfile = lmdb.open(name,map_size=int(1e12))
			
		meanrgb = np.zeros(3) # for mean RGB
		numpixel = 0.0
		mag = 1. * len(files)*10000
				
		imgnames = [os.path.split(name)[1] for name in files]
		
		with dbfile.begin(write=True) as txn:
			for (i, img) in enumerate(imgnames):
				I = misc.imread(os.path.join(imgpath,img[:-4]+sufix))
				#misc.imshow(I)
				if len(I.shape) == 3:
					meanrgb += (np.sum(np.sum(I,0),0) / mag)
					numpixel += I.shape[0]*I.shape[1]
					I = I[:,:,::-1] # convert to BGR
					I = scipy.transpose(I,(2,0,1)) # change to CxHxW
				else:# white and black, normally it is groundtruth
					I = I.reshape((I.shape[0],I.shape[1],1))
					I = I.transpose((2,0,1))
				if I.dtype != typ:
					#print "convert to ", typ
					I = I.astype(typ)
				print i, ':', img,'|', I.dtype,'|',  I.shape,'|',  'min: ', I.min(),'|',  'max: ', I.max()
				form = '{:0>10d}'.format(i) # pay attention the # of images has less than 1e10
				try:
					datum = caffe.io.array_to_datum(I)
					txn.put(form, datum.SerializeToString())
				except:
					print "write " + name + "error"
				
		dbfile.close()
		
		if numpixel != 0.0:
			meanrgb = meanrgb*mag/numpixel
		return meanrgb
		#print "mean value R, G, B: ", meanrgb


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

	
