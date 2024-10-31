import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil


class extractUsefulTime():
    def __init__(self, parentPath):
        
        self.xShift=2.5
        self.yTravel=108
        self.zElevate=5
        self.zOffset=40

        self.x_vel =  100# mm/s
        self.y_vel = 100 # mm/s
        self.z_vel = 5 # mm/s

        self.initTime = 0.15625906
        self.delayTime = 20
        self.offsetTime = self.zOffset/self.z_vel # mounting position to initial position, movement only done in Z-axis.
        self.forwardTime = self.yTravel/self.y_vel
        self.backwardTime = self.yTravel/self.y_vel
        self.shiftTime = self.xShift/self.x_vel
        self.upTime = self.zElevate/self.z_vel
        self.downTime = self.zElevate/self.z_vel
        # self.stopTime = 0.11384
        self.stopTime = 0

        self.numPasses = 28
        self.numForward = 14
        self.numBackward = self.numPasses - self.numForward
        self.parentPath = parentPath
        initCSVpath = os.path.join(self.parentPath,'initTime.csv')
        self.initTimeDF = pd.read_csv(initCSVpath)


    def _makeTimeIntervals(self,initTime):
        t = 0
        t = t + initTime + self.delayTime + self.offsetTime
        time_intervals = []
        t0 = t
        # df = pd.read_csv(self.csvFilepath)
        for _ in range(self.numForward):
            # Useful data
            t = t + self.backwardTime
            time_intervals.append([t0,t])
            # # Not-Useful Data
            t = t + self.shiftTime + self.upTime + self.forwardTime \
                  + self.downTime + 5*self.stopTime
            t0 = t
        # time_intervals = [[time_intervals[0][0], time_intervals[-1][1]]]
        # print(tim)
        return time_intervals
    
    def removeTime(self,):
        # print(time_intervals)
        # exit()
        final_dir = './Datasets/cleaned_'+self.parentPath.split('_')[1]

        try:
            shutil.rmtree(final_dir)
        except:
            pass
        
        os.mkdir(final_dir)
        for class_name in os.listdir(self.parentPath):

            class_path = os.path.join(self.parentPath, class_name)
            os.mkdir(os.path.join(final_dir, class_name))
            if os.path.isdir(class_path):
                for csvFile in os.listdir(class_path):
                    if csvFile.endswith('.csv'):
                        fileNum = os.path.splitext(csvFile)[0]
                        initTime = self.initTimeDF[(self.initTimeDF['Texture']==class_name) & \
                                                   (self.initTimeDF['Iteration']==int(fileNum))]['Initialization Time'].item()
                        
                        time_intervals = self._makeTimeIntervals(initTime)
                        csvFilepath = os.path.join(class_path, csvFile)
                        c = 0
                        df = pd.read_csv(csvFilepath)
                        new_df = df.iloc[0:0,:].copy()
                        for index,row in df.iterrows():
                            if row['Time'] >= time_intervals[c][0] and row['Time'] <= time_intervals[c][1]:
                                new_df.loc[len(new_df)] = row
                            elif row['Time'] >= time_intervals[c][1]:
                                if c+1!=len(time_intervals):
                                    if row['Time'] <= time_intervals[c+1][0]:
                                        c+=1
                        new_df.to_csv(final_dir+'/'+class_name+'/'+csvFile, index=False)
                    exit()
remTime = extractUsefulTime('./Datasets/raw_vel6000')
remTime.removeTime()

def mapPoints():
    pass