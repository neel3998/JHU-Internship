import os
import numpy as np
from config import config

class makeGcode():
    def __init__(self,):
        config.__init__(self,)
        try:
            os.remove(self.gcodefile)
        except:
            pass
    def generate(self):
        x = self.xstart
        y = self.ystart
        z = self.zstart

        file = open(self.gcodefile,'w')
        for text in self.start_text:
            file.write(text + '\n')
        while(x<self.xstart + self.xtravel):

            init_pos = "G1 F"+str(self.feed)+" X"+str(x)+" Y"+str(self.ystart)+" Z"+str(self.zstart)+" E0"
            forwardpass = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart + self.ytravel)+" Z"+str(self.zstart)+" E0"
            x = x + self.xshift
            shiftpass = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart + self.ytravel)+" Z"+str(self.zstart)+" E0"
            backwardpass = "G1 F"+str(self.feed)+" X"+str(x)+" Y"+str(self.ystart)+" Z"+str(self.zstart)+" E0"
            x = x + self.xshift
            
            file.write(init_pos + '\n' + forwardpass + '\n' + shiftpass + '\n'+ backwardpass+'\n')
        for text in self.end_text:
            file.write(text + '\n')
        file.close()
        
        # print(self.start_text)


    

