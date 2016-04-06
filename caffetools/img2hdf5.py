# convert data to hdf5 
# here I focus on associate each image with two mask labels
# **should control the scale of data and label by self for different purposes
# Zizhao @ UF
import os
import numpy as np
import h5py
import glob
import scipy.misc as misc

class img2hdf5:

	def __init__(self, datapath, labelpath, imsz, mean, ifshuffle=True, mdpf=1000):
		self.datapath = datapath
		self.labelpath = labelpath[0]
		if len(labelpath) > 1:
			self.label2path = labelpath[1]
		else:
			self.label2path = False

		self.imsz = imsz
		self.ifshuffle = ifshuffle
		self.mean_value = None;

		self.max_data_per_file = mdpf
		self.mean_value = [];
		self.setmean(mean)

	def writeHDF5(self, savepath, imsufix):

		imlist = glob.glob(os.path.join(self.datapath, '*'+imsufix))
		
		if self.label2path: doublelabel=True
		

		if self.ifshuffle:
			print "-->> shuffling images"
			np.random.shuffle(imlist) # random shuffle data
		# split index to each file
		mnum = self.max_data_per_file
		file_size = int(np.ceil(len(imlist) / float(mnum)))

		file_list = list()
		for i in range(file_size):
			idx = range(i*mnum, min(len(imlist),(i+1)*mnum))
			tmp = [imlist[s] for s in idx]
			file_list.append(tmp)
		cc = 0
		for (bb,blist) in enumerate(file_list): # for each file

			D = np.zeros((len(blist), 3, self.imsz[0], self.imsz[1]), dtype='float32')
			Y = np.zeros((len(blist), 1+doublelabel, self.imsz[0], self.imsz[1]), dtype='float32')

			for (k, item) in enumerate(blist): # for each data
				cc += 1
				print cc, 
				(path, imgname) = os.path.split(item)
				# imgname = imname[:-4]
				img = misc.imread(item)
				img = img - self.mean_value # minus mean

				(h, w , c) = img.shape
				assert(h == self.imsz[0] and w == self.imsz[1])
				img = img[:,:,::-1].astype('float32') # change to BGR
				D[k] = np.transpose(img,(2,0,1)) # change to CxHxW
			    # read labels
				label = misc.imread(os.path.join(self.labelpath, imgname))
				label = label > 0 # scale to 1
				Y[k, 0] = label.reshape((h,w,1)).astype('float32').transpose((2,0,1))
				# the second label
				if self.label2path:
					label2 = misc.imread(os.path.join(self.label2path, imgname))
					label2 = label2 / 255.0
					Y[k, 1] = label2.reshape((h,w,1)).astype('float32').transpose((2,0,1))
					# label = np.concatenate((label, label2), axis=0) # augment the label set

			if not os.path.exists(savepath):
				os.makedirs(savepath)
			print 
			print 'total data shape:', D.shape
			print 'total label shape:', Y.shape
			# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
			# To show this off, we'll list the same data file twice.
			train_filename = os.path.join(savepath, 'train'+str(bb)+'.h5')

			with h5py.File(train_filename, 'w') as f:
			    f['data'] = D
			    f['label'] = Y
			 
			with open(os.path.join(savepath, 'train.txt'), 'a') as f:
			    f.write(train_filename + '\n')
				    # f.write(train_filename + '\n')
			print '-->> write #', bb, "files"
				# HDF5 is pretty efficient, but can be further compressed.
				# comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
				# with h5py.File(test_filename, 'w') as f:
				#     f.create_dataset('data', data=Xt, **comp_kwargs)
				#     f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
				# with open(os.path.join(savapath, 'test.txt'), 'w') as f:
				#     f.write(test_filename + '\n')
	def writeHDF5_singlelabel(self, savepath, imsufix):

		imlist = glob.glob(os.path.join(self.datapath, '*'+imsufix))
		
		if self.label2path: doublelabel=True
		
		if self.ifshuffle:
			print "-->> shuffling images"
			np.random.shuffle(imlist) # random shuffle data
		# split index to each file
		mnum = self.max_data_per_file
		file_size = int(np.ceil(len(imlist) / float(mnum)))

		file_list = list()
		for i in range(file_size):
			idx = range(i*mnum, min(len(imlist),(i+1)*mnum))
			tmp = [imlist[s] for s in idx]
			file_list.append(tmp)
		cc = 0
		for (bb,blist) in enumerate(file_list): # for each file

			D = np.zeros((len(blist), 3, self.imsz[0], self.imsz[1]), dtype='float32')
			Y = np.zeros((len(blist)), dtype='float32')

			for (k, item) in enumerate(blist): # for each data
				cc += 1
				(path, imgname) = os.path.split(item)
				# imgname = imname[:-4]
				img = misc.imread(item)
				img = img - self.mean_value # minus mean

				(h, w , c) = img.shape
				assert(h == self.imsz[0] and w == self.imsz[1])
				img = img[:,:,::-1].astype('float32') # change to BGR
				D[k] = np.transpose(img,(2,0,1)) # change to CxHxW
				# read labels
				label = float(imgname[0])
				Y[k] = label

			if not os.path.exists(savepath):
				os.makedirs(savepath)
			print 
			print 'total data shape:', D.shape
			print 'total label shape:', Y.shape
			# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
			# To show this off, we'll list the same data file twice.
			train_filename = os.path.join(savepath, 'train'+str(bb)+'.h5')

			with h5py.File(train_filename, 'w') as f:
			    f['data'] = D
			    f['label'] = Y
			 
			with open(os.path.join(savepath, 'train.txt'), 'a') as f:
			    f.write(train_filename + '\n')
				    # f.write(train_filename + '\n')
			print '-->> write #', bb, "files"
				# HDF5 is pretty efficient, but can be further compressed.
				# comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
				# with h5py.File(test_filename, 'w') as f:
				#     f.create_dataset('data', data=Xt, **comp_kwargs)
				#     f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
				# with open(os.path.join(savapath, 'test.txt'), 'w') as f:
				#     f.write(test_filename + '\n')
	def setmean(self, value):
		self.mean_value = value # **Note**: mean value in RGB order
		self.mean_value = np.asarray(self.mean_value)
		self.mean_value = self.mean_value.reshape((1,1,3))

	def checkHDF5(self, path, savepath='./'):
		f = h5py.File(path,'r')

		num = f['data'].shape[0]
		for i in range(num):
			data = f['data'][i] 
			data = data.transpose((1,2,0))[:,:,::-1] + self.mean_value
			(h,w,c) = data.shape

			label1 = f['label'][i,0]
			label1 = label1.reshape((h, w, 1))
			label1 = np.tile(label1, (1, 1, 3))
			label1 = label1.astype('uint8')*255

			label2 = f['label'][i,1]
			label2 = label2.reshape((h, w, 1))
			label2 = np.tile(label2, (1, 1, 3))
			label2 = label2.astype('uint8')*255
			img = np.concatenate((data, label1, label2), axis=1)

			misc.imsave(os.path.join(savepath, str(i)+'.png'), img)

if __name__ == "__main__":

	dpath = '/home/zizhaozhang/caffe/github/end2end/data/MUSCLE-SMALL/smallset/images/'
	labelpath = '/home/zizhaozhang/caffe/github/end2end/data//MUSCLE-SMALL/smallset/groundTruth/contours/'
	label2path = '/home/zizhaozhang/caffe/github/end2end/data//MUSCLE-SMALL/smallset/groundTruth/seg/'
	imsz = (300,300)
	handle =  img2hdf5(dpath, [labelpath, label2path], imsz, ifshuffle=True)
	mean = [0,0,0]
	handle.setmean(mean)
	savepath = ''
