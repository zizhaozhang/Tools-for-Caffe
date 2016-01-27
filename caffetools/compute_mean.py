import numpy as np
import scipy
import scipy.misc as misc
import glob
import os

def compute_mean(img_path, sufix):
	namelist = glob.glob(img_path+'*'+sufix)
	if len(namelist) == 0:
		print "no image found."
		return 
		
	# mean_value = np.zeros()
	for (idx, item) in enumerate(namelist):
		im = misc.imread(item)
		print '#', idx, ' size: ', im.shape, 'per-image mean: ', np.mean(np.mean(im,axis=0), axis=0)
		if idx == 0:
			mean_value = np.zeros(im.shape)

		mean_value += im 

	mean_value  = mean_value / len(namelist)
	mean_value = np.mean(np.mean(mean_value,axis=0), axis=0)
	
	print 'R:{0}, G:{1}, B:{2}'.format(mean_value[0], mean_value[1], mean_value[2])
	return mean_value
