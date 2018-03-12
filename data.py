from scipy.io import loadmat
import os
import numpy as np
import pickle
from random import shuffle,seed
from StringIO import StringIO
from scipy.signal import resample

def to_onehot(labels):
    unique_labels = list(set(labels))
    corrected_labels = map(lambda x: unique_labels.index(x),labels)
    y = np.zeros((len(labels),len(unique_labels)))
    y[range(len(labels)),corrected_labels] = 1
    return y

def data_shuffle(x,y,random_state=None,subj_indices=None):
    seed(random_state)
    d_len = len(y)
    sh_data = range(d_len)
    shuffle(sh_data)
    new_y = np.zeros_like(y)
    for i in range(d_len):
        new_y[i,...] = y[sh_data[i],...]
    new_x = np.zeros(x.shape)
    for i in range(d_len):
        new_x[i,...] = x[sh_data[i],...]
    if subj_indices is not None:
        new_subj_indices=np.zeros_like(subj_indices)
        for i in range(d_len):
            new_subj_indices[i, ...] = subj_indices[sh_data[i], ...]
        return new_x, new_y,subj_indices
    return new_x,new_y

class Data:
    def __init__(self,path_to_data,start_epoch,end_epoch,sample_rate=500):
        self.start_epoch = start_epoch  # seconds
        self.end_epoch = end_epoch
        self.sample_rate = sample_rate
        self.path_to_data = path_to_data
        if len(path_to_data.split('/')[-1]) == 0:
            self.exp_name = path_to_data.split('/')[-2]
        else:
            self.exp_name = path_to_data.split('/')[-1]

# class DataProcessExperiment(Data):
#     def __init__(self,path_to_data):
#         start_epoch = -1.5 #seconds
#         end_epoch = 1 #seconds
#         Data.__init__(self,path_to_data,start_epoch)
#
#     def get_data(self,shuffle=False,start_window=0.200,end_window=0.500):
#         """
#         Returns:
#             A tuple of 2 numpy arrays: data (Trials x Channels x Time) and labels
#         """
#         data_info = loadmat(os.path.join(self.path_to_data,'events.mat'))['events']['field_type'][0][0]
#         start_window_ind = int((start_window - self.start_epoch)*self.sample_rate)
#         end_window_ind = int((end_window - self.start_epoch)*self.sample_rate)
#         indexes = [(i,(str[0][0] in ['ball','field'])) for i,str in enumerate(data_info) if str[0][0] in ['ball','ball_nT','field','field_nT']]
#         indexes,labels = map(lambda x:list(x),zip(*indexes))
#         data = loadmat(os.path.join(self.path_to_data,'eeg_epochs.mat'))['eeg_epochs'].transpose(2,0,1)
#         if shuffle:
#             return data_shuffle(data[indexes,start_window_ind:end_window_ind,:],labels)
#         return data[indexes,start_window_ind:end_window_ind,:],labels

class DataBuildClassifier(Data):
    def __init__(self, path_to_data):
        '''
        :param path_to_data: string, path to folder with all experiments
        '''
        start_epoch = -0.5 #seconds
        end_epoch = 1#seconds
        #super(DataBuildClassifier, self).__init__(path_to_data,start_epoch,end_epoch)
        Data.__init__(self, path_to_data,start_epoch,end_epoch)

    def _baseline_normalization(self,X,baseline_window=()):
        bl_start = int((baseline_window[0] - self.start_epoch) * self.sample_rate)
        bl_end = int((baseline_window[1] - self.start_epoch) * self.sample_rate)

        return X[:,bl_start:bl_end,:].mean(axis=1)
    
    def _reasample(self, X, y, reasample_to):
            duration = self.end_epoch - self.start_epoch
            downsample_factor = X.shape[1]/(resample_to * duration)
            print 'rrr'
            return resample(X,up=1., down=downsample_factor, npad='auto',axis=1), y
        
    def get_data(self,subjects,shuffle=False,random_state=None,windows=None,baseline_window=(),resample_to=None):
        '''

        :param subjects: list subject's numbers, wich data we want to load
        :param shuffle: bool
        :param windows: list of tuples. Each tuple contains two floats - start and end of window in seconds
        :param baseline_window:
        :return: Dict. {Subject_number:tuple of 2 numpy arrays: data (Trials x Time x Channels) and labels}
        '''
        res={}
        for subject in subjects:

            eegT = loadmat(os.path.join(self.path_to_data,str(subject),'eegT.mat'))['eegT']
            eegNT = loadmat(os.path.join(self.path_to_data,str(subject),'eegNT.mat'))['eegNT']
            X = np.concatenate((eegT,eegNT),axis=-1).transpose(2,0,1)
            if len(baseline_window):
                baseline = self._baseline_normalization(X,baseline_window)
                baseline = np.expand_dims(baseline,axis=1)
                X = X - baseline
            y = np.hstack((np.ones(eegT.shape[2]),np.zeros(eegNT.shape[2])))
            #y = np.hstack(np.repeat([[1,0]],eegT.shape[2],axis=0),np.repeat([[0,1]],eegT.shape[2],axis=0))
            
            if (resample_to is not None) and (resample_to != self.sample_rate):
                X, y = _reasample(X, y, reasample_to)
            time_indices=[]
            if windows is not None:
                for win_start,win_end in windows:
                    start_window_ind = int((win_start - self.start_epoch)*self.sample_rate)
                    end_window_ind = int((win_end - self.start_epoch)*self.sample_rate)
                    time_indices.extend(range(start_window_ind,end_window_ind))
                X,y = X[:,time_indices,:],y

            if shuffle:
                X,y = data_shuffle(X,y,random_state=random_state)
            res[subject]=(X,y)
        return res





if __name__ == '__main__':
    data = DataBuildClassifier('/home/likan_blk/BCI/NewData/').get_data([33],shuffle=True,
                                                                               windows=[(0.2, 0.5)],baseline_window=(0.2, 0.3))
    print 1
