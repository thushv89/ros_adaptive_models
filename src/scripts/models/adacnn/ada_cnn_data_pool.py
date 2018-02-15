
import numpy as np
from collections import Counter
from sklearn.utils import shuffle

class Pool(object):
    '''
    This class is used to maintain a pool of data which will be used
    from time to time for finetuning the network
    '''
    def __init__(self,**params):

        self.image_size = params['image_size']
        self.num_channels = params['num_channels']
        self.num_env = params['num_env']
        self.assert_test = params['assert_test']
        self.size = params['size']
        self.num_labels = params['num_labels']
        self.position = 0
        self.dataset = np.empty((self.size,self.image_size[0],self.image_size[1],params['num_channels']),dtype=np.float32)
        self.labels = np.empty((self.size,params['num_labels']),dtype=np.float32)
        self.env_ids = np.empty((self.size,1),dtype=np.float32)
        self.batch_size = params['batch_size']
        self.filled_size = 0

    def add_all_from_ndarray(self,data,labels,env_ids):
        '''
        Add all the data to the pool
        :param data:
        :param labels:
        :return:
        '''
        add_size = data.shape[0]

        if self.position + add_size <= self.size -1:
            self.dataset[self.position:self.position+add_size,:,:,:] = data
            self.labels[self.position:self.position+add_size,:,:,:] = labels
            self.env_ids[self.position:self.position+add_size,:] = env_ids
        else:
            overflow = (self.position + add_size) % (self.size-1)
            end_chunk_size = self.size - (self.position + 1)

            assert overflow + end_chunk_size == add_size
            # Adding till the end
            self.dataset[self.position:,:,:,:] = data[:end_chunk_size+1,:,:,:]
            self.labels[self.position:,:] = labels[:end_chunk_size+1,:]
            self.env_ids[self.position,:] = env_ids[:end_chunk_size+1,:]

            # Starting from the beginning for the remaining
            self.dataset[:overflow,:,:,:] = data[end_chunk_size:,:,:,:]
            self.labels[:overflow,:] = labels[end_chunk_size:,:]
            self.env_ids[:overflow,:] = env_ids[end_chunk_size:,:]

        if self.assert_test:
            assert np.all(self.dataset[self.position,:,:,:].flatten()== data[0,:,:,:].flatten())

        if self.filled_size != self.size:
            self.filled_size = min(self.position+add_size+1,self.size)

        self.position = (self.position+add_size)%self.size

    def add_hard_examples(self,data,labels,env_ids,loss,fraction,randomize):
        '''
        This method will add only the hard examples to the pool
        Hard examples are the ones that were not correctly classified
        :param data: full data batch
        :param labels: full label batch
        :param loss: loss vector for the batch
        :param fraction: fraction of data we want
        :return: None
        '''

        if not randomize:
            hard_indices = np.argsort(loss).flatten()[::-1][:int(fraction * self.batch_size)]
        else:
            hard_indices = np.random.choice(np.arange(self.batch_size),size=int(fraction*self.batch_size))

        add_size = hard_indices.size

        # if position has more space for all the hard_examples
        if self.position + add_size <= self.size - 1:
            self.dataset[self.position:self.position+add_size,:,:,:] = data[hard_indices,:,:,:]
            self.labels[self.position:self.position+add_size,:] = labels[hard_indices,:]
            self.env_ids[self.position:self.position+add_size,:] = env_ids[hard_indices,:]
        else:
            overflow = (self.position + add_size) % (self.size-1)
            end_chunk_size = self.size - (self.position + 1)

            assert overflow + end_chunk_size == add_size
            # Adding till the end
            self.dataset[self.position:,:,:,:] = data[hard_indices[:end_chunk_size+1],:,:,:]
            self.labels[self.position:,:] = labels[hard_indices[:end_chunk_size+1],:]
            self.env_ids[self.position:,:] = env_ids[hard_indices[:end_chunk_size+1],:]

            # Starting from the beginning for the remaining
            self.dataset[:overflow,:,:,:] = data[hard_indices[end_chunk_size:],:,:,:]
            self.labels[:overflow,:] = labels[hard_indices[end_chunk_size:],:]
            self.env_ids[:overflow,:] = env_ids[hard_indices[end_chunk_size:],:]

        if self.assert_test:
            assert np.all(self.dataset[self.position,:,:,:].flatten()== data[hard_indices[0],:,:,:].flatten())

        if self.filled_size != self.size:
            self.filled_size = min(self.position+add_size+1,self.size)

        self.position = (self.position+add_size)%self.size

    def get_position(self):
        return self.position

    def get_size(self):
        return self.filled_size

    def get_pool_data(self,do_shuffle):
        if do_shuffle:
            self.dataset, self.labels, self.env_ids = shuffle(self.dataset, self.labels,self.env_ids)
            return self.dataset,self.labels
        else:
            return (self.dataset,self.labels)

    def get_class_distribution(self):
        class_count = Counter(np.argmax(self.labels[:self.filled_size,:],axis=1).flatten())
        dist_vector = []
        for lbl in range(self.num_labels):
            if lbl in class_count:
                dist_vector.append(class_count[lbl]*1.0/self.filled_size)
            else:
                dist_vector.append(0.0)
        return dist_vector

    def get_env_sample_count_string(self):
        class_count = Counter(self.env_ids[:self.filled_size, :].flatten())
        count_str = ''
        for ei in range(self.num_env):
            val = class_count[ei] if ei in class_count else 0.0
            count_str += str(val)+','

        return count_str

    def reset_pool(self):
        self.position = 0
        self.dataset = np.empty((self.size,self.image_size[0],self.image_size[1],self.num_channels),dtype=np.float32)
        self.labels = np.empty((self.size,self.num_labels),dtype=np.float32)
        self.env_ids = np.empty((self.size, 1), dtype=np.float32)
        self.filled_size = 0

