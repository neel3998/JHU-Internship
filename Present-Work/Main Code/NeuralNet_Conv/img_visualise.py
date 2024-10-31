import numpy as np
import ast
import cv2
import pandas as pd
from models.neuronCluster import neuronCluster
import matplotlib.pyplot as plt

class visualise():
    def __init__(self):
        self.seq_len = 128
        self.V_df = pd.read_csv('3.csv')
        self.Z_df = pd.read_csv('Train_Z.csv')

        self.clusters = []
        for _ in range(16):
            self.clusters.append(neuronCluster())

    def get_dict(self, index):
        data_dict = {}
        # self.seq_len = 128

        check = 0
        t_dash = index
        classname = self.V_df[index:index+1]['Class'].item()
        iter_no = self.V_df[index:index+1]['iter_No'].item()
        pass_no = self.V_df[index:index+1]['Pass_No'].item()

        for i in range(128):
            m = index - i

            if m == 0:
                t_dash = m
                check = 1
                break
            if (self.V_df[m:m+1]['Class'].item() != classname or \
                self.V_df[m:m+1]['iter_No'].item() != iter_no or \
                self.V_df[m:m+1]['Pass_No'].item() != pass_no):
                t_dash = m
                check = 1
                break

        for key in self.V_df.keys():
            if key.isnumeric():
                if check:
                    temp_v = np.array([2.44]*(self.seq_len - index + t_dash))
                    temp_z = np.array([0]*(self.seq_len - index + t_dash))

                    temp_v = np.hstack((temp_v, np.array(self.V_df[key].to_list()[t_dash+1:index+1])))
                    temp_z = np.hstack((temp_z, np.array(self.Z_df[key].to_list()[t_dash+1:index+1])))

                    
                    cluster_v = (self.clusters[int(key)].runCluster(temp_v) + 50)/131
                    data_dict['v_'+key] = cluster_v.transpose(1,0)
                    data_dict['z_'+key] = (temp_z + 2.5)/5

                else:
                    temp_v = np.array(self.V_df[key].to_numpy()[index-self.seq_len+1:index+1])
                    temp_z = np.array(self.Z_df[key].to_numpy()[index-self.seq_len+1:index+1])


                    cluster_v = (self.clusters[int(key)].runCluster(temp_v) + 50)/131
                    data_dict['v_'+key] = cluster_v.transpose(1,0)
                    # data_dict['z_'+key] = (self.Z_df[key][index:index+1].item() + 2.5)/5
                    data_dict['z_'+key] = (temp_z + 2.5)/5

        
        return data_dict

    # def image_remodel(self,):

v = visualise()
v.seq_len = 500
a = v.get_dict(4200)
plt_dat = a['v_6'].transpose(1,0)
plt.plot(plt_dat[0])
plt.plot(plt_dat[1])
plt.plot(plt_dat[2])

st = a['v_0'].reshape(v.seq_len,1,3)
z_st = a['z_0'].reshape(v.seq_len,1,1)
for i in range(5):
    st = np.hstack((st,st))
    z_st = np.hstack((z_st,z_st))


for i in range(1,16):
    temp = a['v_'+str(i)].reshape(v.seq_len,1,3)
    z_temp = a['z_'+str(i)].reshape(v.seq_len,1,1)

    for _ in range(5):
        temp = np.hstack((temp,temp))
        z_temp = np.hstack((z_temp,z_temp))
    st = np.hstack((st,temp))
    z_st = np.hstack((z_st,z_temp))


# print(z_st)
cv2.imshow('i',st)
# cv2.imshow('z',z_st)

plt.show()

cv2.waitKey(0)