# from gcode import makeGcode

# g = makeGcode()
# g.generate()
import serial

ser = serial.Serial('USB Serial', 115200)
ser.write(str.encode("G1 F100 X0 Y0 Z20 E0"))
ser.close()