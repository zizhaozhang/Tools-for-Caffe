# waiting to be finished
import h5py
import caffe
import numpy as np
path = '/home/zizhaozhang/cell_data/large_train_SRF/images/train/'
def write(shape,name='test_train_hdf5'):
    f = h5py.File(name,'w')
    f.create_dataset("Image",shape,dtype=np.float32)
    f.create_dataset("Mask",shape,dtype=np.int32)
           
if __name__ == "__main__":
    pass
    #write(name)
