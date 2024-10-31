import os, sys, glob
sys.path.append('./framework/libraries/general')
sys.path.append('./framework/libraries/neuromorphic')

import time
from threading import Thread, Lock #control access in threads
from copy import copy #useful for copying data structures

from frameworks.libraries.neuromorphic.tactileboard import * #tactile board library
from frameworks.libraries.general.threadhandler import ThreadHandler #manage threads

def init_tactile():
    global tactileBoard
    thMain = ThreadHandler(tactile_update)
    tactileBoard = TactileBoard('COM5',_sensitivity=TBCONSTS.HIGH_SENS)
    tactileBoard.start()
    thMain.start()
    # tactileBoard.startCalibration(1000) #ask for 1000 samples --> 1s calibration
    # time.sleep(2) #make sure that you wait for calibration to be finished
    # tactileBoard.loadCalibration()
    # tactileBoard.useCalib = True
    print('--TACTILE READY--')

def tactile_update():
    global tactileBoard, dataQueue, fileHandler, taxel_array
    taxel_array=[]
    dataQueue = tactileBoard.getData()
    if (len(dataQueue)>0):
        n = len(dataQueue)
        for k in range(n):
            data = dataQueue.popleft()
            for i in range(TBCONSTS.NROWS): #number of rows
                for j in range(TBCONSTS.NCOLS):
                    taxel = data[0][i][j] #this is normalized value
                    rawv = tactileBoard.conv2raw(taxel,0,i,j)
                    fileHandler.write(str(rawv) + ' ') #write the normalized value to the file
                  #  print(rawv)
                    taxel_array.append(rawv)
            fileHandler.write('\n') #new line
    time.sleep(0.001)
    return taxel_array

def main():
    global fileHandler

    init_tactile()

    filename = "tactile_test.txt"
    fileHandler = open(filename,'w')

    # tactile_update()
    # time.sleep(10)

    # fileHandler.close()
    # print('--Close file--')


    t1=time.time()
    t2=t1
    while(t2-t1<10):
        isinstat_data=tactile_update()
      #  print(isinstat_data)
        t2=time.time()
        print(t2-t1)
      #  if(t2-t1>5):
    fileHandler.close()


if __name__ == '__main__':
	main()
