#########################
# Code by - Rwik and Neel --- @IITGN and collaboration with @JHU
#########################
# This is a custom g-code .txt file generator for palpation of a finger over the texture. 
# The movements considered in the Gcode are always parallel to one of the axes.
'''
Usage:
1. Make an instance of the class makeGcode()
2. Call the function generateGcodefile. 
   NOTE: use ONLY typee -> B and an appropriate delay according to the mounting and un-mounting times.
   Other function are experimental and wont give good results.
3. Once the function is compeleted, check "fingergcode.txt".
4. Convert the file by saving it as a .gcode file.
5. Viola! Enjoy!

'''
import os
import numpy as np
from .config import config
import serial
import time

class makeGcode():
    def __init__(self,):
        config.__init__(self,)
        try:
            os.remove(self.gcodefile)
        except:
            pass
        self.isextraCode = 1

    def generateGcodefile(self, type = 'B', delay = 0):
        '''
        This function generates the Gcode file in a .txt format.
        User has to save it in .gcode format manually.
        # ----------
        Input:
        type --> type of palplation. 
                "B" -- Only Backward palpation i.e. in forward pass, there 
                       will be a z offset and finger wont touch the specimen.
                       (Default)
                "FB" -- front and backward i.e. the finger will touch the specimen 
                        in both forward and backward pass.
        
        delay --> Time delay in milliseconds to help user mount and un-mount the finger.
        # ----------
        Output:
        Writes the gcode into a file called "fingergcode.txt"
        '''

        # # Front and Back
        # if type == "FB":
        #     x = self.xstart
        #     y = self.ystart
        #     z = self.zstart

        #     file = open(self.gcodefile,'w')
        #     for text in self.start_text:
        #         file.write(text + '\n')
        #     while(x<self.xstart + self.xtravel):

        #         init_pos = "G1 F"+str(self.feed)+" X"+str(x)+" Y"+str(self.ystart)+" Z"+str(self.zstart)+" E0"
        #         forwardpass = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart + self.ytravel)+" Z"+str(self.zstart)+" E0"
        #         x = x + self.xshift
        #         shiftpass = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart + self.ytravel)+" Z"+str(self.zstart)+" E0"
        #         backwardpass = "G1 F"+str(self.feed)+" X"+str(x)+" Y"+str(self.ystart)+" Z"+str(self.zstart)+" E0"
        #         x = x + self.xshift
                
        #         file.write(init_pos + '\n' + forwardpass + '\n' + shiftpass + '\n'+ backwardpass+'\n')
        #     for text in self.end_text:
        #         file.write(text + '\n')
        #     file.close()


        # Backward only
        if type == "B":
            x = self.xstart
            y = self.ystart
            z = self.zstart

            file = open(self.gcodefile,'w')

            #Next 3 lines Not required
            if self.isextraCode:
                for text in self.start_text:
                    file.write(text + '\n')

            # Setting the metric system to mm
            file.write("G21\n")
            # Initial position
            file.write('G1 F'+str(self.feed)+" X"+str(x) +" Y"+str(self.ystart)+" Z"+str(self.zstart + 40)+" E0\n")
            # Mounting time and placing the specimen.
            file.write("G4 P"+str(delay)+'\n')            

            # Start the palpation 
            while(x<self.xstart + self.xtravel):

                startpos = "G1 F"+str(self.feed)+" X"+str(x)+" Y"+str(self.ystart)+" Z"+str(self.zstart)+" E0"
                forwardpass = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart + self.ytravel)+" Z"+str(self.zstart)+" E0" 
                x = x + self.xshift
                shiftpass = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart + self.ytravel)+" Z"+str(self.zstart)+" E0"
                shiftUp = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart+ self.ytravel)+" Z"+str(self.zstart + self.z_elevate)+" E0"
                backwardpass = "G1 F"+str(self.feed)+" X"+str(x)+" Y"+str(self.ystart)+" Z"+str(self.zstart + self.z_elevate)+" E0"
                shiftDown = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart)+" Z"+str(self.zstart)+" E0"
                file.write(startpos + '\n' + forwardpass + '\n' + shiftpass + '\n'+ shiftUp + '\n'+backwardpass+'\n'+ shiftDown+'\n')

            file.write('G1 F'+str(self.feed)+" X"+str(x) +" Y"+str(self.ystart)+" Z"+str(self.zstart + 40)+" E0\n")
            file.write('G1 F'+str(self.feed)+" X"+str(self.xstart) +" Y"+str(self.ystart)+" Z"+str(self.zstart + 40)+" E0\n")
            # End
            
            # Next 3 lines Not required
            if self.isextraCode:
                for text in self.end_text:                
                    file.write(text + '\n')
            file.close()


    # EXPERIMENTAL CODE DONT USE ***********
    def generateGcodeSerial(self,type, delay):
        # Backward only
        if type == "B":
            x = self.xstart
            y = self.ystart
            z = self.zstart
            ser = serial.Serial('COM7', 115200)
            time.sleep(5)
            # file = open(self.gcodefile,'w')
            # for text in self.start_text:
            #     file.write(text + '\n')
            # Delay counter
            c = 0
            while(x<self.xstart + self.xtravel):

                init_pos = "G1 F"+str(self.feed)+" X"+str(x)+" Y"+str(self.ystart)+" Z"+str(self.zstart)+" E0\r\n"
            
                forwardpass = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart + self.ytravel)+" Z"+str(self.zstart)+" E0\r\n" 
                x = x + self.xshift
                shiftpass = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart + self.ytravel)+" Z"+str(self.zstart)+" E0\r\n"
                shiftUp = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart+ self.ytravel)+" Z"+str(self.zstart + self.z_elevate)+" E0\r\n"
                backwardpass = "G1 F"+str(self.feed)+" X"+str(x)+" Y"+str(self.ystart)+" Z"+str(self.zstart + self.z_elevate)+" E0\r\n"
                shiftDown = "G1 F"+str(self.feed)+" X"+str(x)+ " Y"+str(self.ystart)+" Z"+str(self.zstart)+" E0\r\n"

                commands = [init_pos, forwardpass, shiftpass, shiftUp, backwardpass, shiftDown]

                if delay!=0:
                    if c == 0:
                        # Mounting Time
                        mount_pos = "G1 F"+str(self.feed)+" X"+str(x)+" Y"+str(self.ystart)+" Z"+str(self.zstart + 40)+" E0\r\n"
                        ser.write(str.encode(mount_pos))
                        ser.write(str.encode("G4 P"+str(delay)+'\r\n'))
                        # file.write(mount_pos + '\n'+ "G4 P"+str(delay)+ init_pos +'\n'+ '\n' + forwardpass + '\n' + shiftpass + '\n'+ shiftUp + '\n'+backwardpass+'\n'+ shiftDown+'\n')
                        c = 1
                        print("Neel")
                for command in commands:
                    ser.write(str.encode(command))
                    # time.sleep(1)
                    print('reading:')
                    r = ser.readline().decode('ascii')
                    while r=='echo:busy: processing\n':
                        print(r)
                        r = ser.readline().decode('ascii')
                    print(r)

                    # print(ser.readline().decode('ascii'))
                    # time.sleep(1)
                    # print("a")
            print("oafhdoijdsfoij")
            # Unmounting Time
            if c == 1:
                ser.write(str.encode(mount_pos))
                ser.write(str.encode("G4 P"+str(delay)+'\r\n'))
                for command in commands:
                    ser.write(str.encode(command))
                # file.write(mount_pos + '\n'+ "G4 P"+str(delay)+ init_pos +'\n'+ '\n' + forwardpass + '\n' + shiftpass + '\n'+ shiftUp + '\n'+backwardpass+'\n'+ shiftDown+'\n')
            
            # for text in self.end_text:                
            #     file.write(text + '\n')
            # file.close()
        


if __name__=="__main__":
    g = makeGcode()

    # Set the offsets, taking in considerations 
    # like mount holder height, mount holder position etc. 

    xoffset = 8.5 #8.5 mm
    yoffset = 44.07 #94.069  mm
    zoffset = 81.31 #81.307 mm

    g.xstart = g.xstart + xoffset
    g.ystart = g.ystart + yoffset 
    g.zstart = g.zstart + zoffset 
    delay = 20000 # msec
    g.generateGcodefile('B', delay)
    

    

