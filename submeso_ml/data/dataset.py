from torch.utils.data import Dataset
import numpy as np
import random
import torch


class SubmesoDataset(Dataset):
    def __init__(self,input_features = ['grad_B','FCOR', 'HML', 'TAU',
              'Q', 'HBL', 'div', 'vort', 'strain'],res='1_4',loc_num=0,seed=123,train_split=0.8):
        super().__init__()
        self.seed=seed
        self.train_split=train_split
        self.input_features=input_features
        self.path= '/scratch/ab10313/pleiades/NN_data_'+'%s' % res+'/'
        self.loc = loc_num

        # load features for input
        in_features=[]
        for input_feature in self.input_features:
            in_features.append(np.load(self.path+'%s.npy' % input_feature))
        # x input
        self.x = torch.from_numpy(np.stack(in_features,axis=1)).float()

        # load output
        WB_sg = np.load(self.path+'WB_sg.npy')

        # y output
        self.y = torch.from_numpy(np.tile(WB_sg,(1,1,1,1)).reshape(WB_sg.shape[0],1,WB_sg.shape[1],WB_sg.shape[2])).float()

        self._get_split_indices()
        self._norm_factors()
        self._seasons_JFM_JAS()
        self._seasons_winter_summer()
        self._locations()
        self._norm_data()

    def _get_split_indices(self):
        """ obtain a set of train and test indices """

        # randomnly generate train, test and validation time indecies
        time_ind = self.x.shape[0]
        rand_ind = np.arange(time_ind)
        rand_seed = self.seed
        random.Random(rand_seed).shuffle(rand_ind)
        self.train_ind, self.test_ind =  rand_ind[:round(self.train_split*time_ind)], rand_ind[round((self.train_split)*time_ind):]

        # sort test_ind
        self.train_ind = np.sort(self.train_ind)


    def _seasons_JFM_JAS(self):
        """ obtain a set of JAS and JFM indices """
        #season indecies
        JAS_ind_min = 577
        JAS_ind_max = 762
        JFM_ind_min = 213
        JFM_ind_max = 396

        self.JFM_ind = np.empty(self.x.shape[0])
        self.JFM_ind[:] = np.nan
        self.JAS_ind = np.empty(self.x.shape[0])
        self.JAS_ind[:] = np.nan


        for i in range(12):
            for j in range(846):
                if JAS_ind_min<j & j<JAS_ind_max:
                    self.JAS_ind[i*846+j] = i*846+j

                elif JFM_ind_min<j & j<JFM_ind_max:
                    self.JFM_ind[i*846+j] = i*846+j


        time_ind = np.arange(self.x.shape[0])
        self.train_JAS_ind = time_ind[np.isnan(self.JAS_ind)]
        self.test_JAS_ind = time_ind[~np.isnan(self.JAS_ind)]
        rand_seed = self.seed
        random.Random(rand_seed).shuffle(self.train_JAS_ind)

        self.train_JFM_ind = time_ind[np.isnan(self.JFM_ind)]
        self.test_JFM_ind = time_ind[~np.isnan(self.JFM_ind)]
        rand_seed = self.seed
        random.Random(rand_seed).shuffle(self.train_JFM_ind)


    def _seasons_winter_summer(self):
        """ obtain a set of JAS and JFM indices """
        #season indecies
        JAS_ind_min = 577
        JAS_ind_max = 762
        JFM_ind_min = 213
        JFM_ind_max = 396

        self.winter_ind = np.empty(self.x.shape[0])
        self.winter_ind[:] = np.nan
        self.summer_ind = np.empty(self.x.shape[0])
        self.summer_ind[:] = np.nan

        for i in range(12):
        # northern hemisphere
            if i==3 or i==4 or i==6 or i==10:
                for j in range(846):
                    if JAS_ind_min<j & j<JAS_ind_max:
                        self.summer_ind[i*846+j] = i*846+j
                    elif JFM_ind_min<j & j<JFM_ind_max:
                        self.winter_ind[i*846+j] = i*846+j
        # southern hemisphere
            elif i==1 or i==2 or i==7 or i==8 or i==9:
                for j in range(846):
                    if JAS_ind_min<j & j<JAS_ind_max:
                        self.winter_ind[i*846+j] = i*846+j
                    elif JFM_ind_min<j & j<JFM_ind_max:
                        self.summer_ind[i*846+j] = i*846+j

        time_ind = np.arange(self.x.shape[0])
        self.train_winter_ind = time_ind[np.isnan(self.winter_ind)]
        self.test_winter_ind = time_ind[~np.isnan(self.winter_ind)]
        rand_seed = self.seed
        random.Random(rand_seed).shuffle(self.train_winter_ind)

        self.train_summer_ind = time_ind[np.isnan(self.summer_ind)]
        self.test_summer_ind = time_ind[~np.isnan(self.summer_ind)]
        rand_seed = self.seed
        random.Random(rand_seed).shuffle(self.train_summer_ind)




    def _locations(self):
        """ obtain location indices """
        #locations
        self.location_ind = np.zeros(self.x.shape[0])

        for i in range(12):
            for j in range(846):
                self.location_ind[i*846+j] = i

        time_ind = np.arange(self.x.shape[0])
        self.train_loc_ind = time_ind[~(self.location_ind==self.loc)]
        self.test_loc_ind = time_ind[(self.location_ind==self.loc)]
        rand_seed = self.seed
        random.Random(rand_seed).shuffle(self.train_loc_ind)


    def _norm_factors(self):
        """ load global noramlization factors: mean and std """
        self.y_mean = torch.from_numpy(np.load(self.path+'WB_sg_mean.npy')*np.ones(self.y.shape)).float()
        self.y_std = torch.from_numpy(np.load(self.path+'WB_sg_std.npy')*np.ones(self.y.shape)).float()

        # load mean and std from features for input
        std_in_features=[]
        mean_in_features=[]
        for input_feature in self.input_features:
            mean_in_features.append(np.load(self.path+'%s_mean.npy' % input_feature)*np.ones((self.x.shape[0],self.x.shape[2],self.x.shape[3])))
            std_in_features.append(np.load(self.path+'%s_std.npy' % input_feature)*np.ones((self.x.shape[0],self.x.shape[2],self.x.shape[3])))

        self.x_mean = torch.from_numpy(np.stack(mean_in_features,axis=1)).float()
        self.x_std = torch.from_numpy(np.stack(std_in_features,axis=1)).float()


    def _norm_data(self):
        """ normalize inputs and output to global normalization factors"""
        self.x_norm = (self.x - self.x_mean)/self.x_std
        self.y_norm = (self.y - self.y_mean)/self.y_std


    def __getitem__(self,idx):
        return (self.x_norm[idx],self.y_norm[idx])



