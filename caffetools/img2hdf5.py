# Write image in path with multiple 
import os
import numpy as np
import h5py
import glob
import scipy.misc as misc

class img2hdf5:

	def __init__(self, datapath, labelpath, imsz):
		self.datapath = datapath
		self.labelpath = labelpath[0]
		if len(labelpath) > 1:
			self.label2path = labelpath[1]

		self.imsz = imsz

	def writeHDF5(self, savapath):

		imlist = glob.glob(os.path.join(self.dpath, '*jpg'))

		for item in imlist:
			(path, imgname) = op.path.split(item)
			# imgname = imname[:-4]
			img = misc.imread(item)
			(h,w) = img.shape
			if len(img.shape) == 3: 
				img = img[:,:,::-1].astype(float32) # change to BGR
				img = np.transpose(I,(2,0,1)) # change to CxHxW

				# read labels
			label = misc.imread(os.path.join(self.labelpath, imgname))
			label.reshape((h,w,1)).astype(float32).transpose((2,0,1))
			


           
if __name__ == "__main__":
  
	dpath = '/home/zizhaozhang/caffe/github/end2end/data/MUSCLE-SMALL/smallset/images/'
	labelpath = '/home/zizhaozhang/caffe/github/end2end/data//MUSCLE-SMALL/smallset/groundTruth/contours/'
	label2path = '/home/zizhaozhang/caffe/github/end2end/data//MUSCLE-SMALL/smallset/groundTruth/seg/'

	handle =  img2hdf5(dpath, [labelpath, label2path])
