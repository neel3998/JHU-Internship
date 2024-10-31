import serial
ser = serial.Serial('COM7', 115200)
ser.write(str.encode("G1 F2000 X0 Y0 Z90 E0\r\n"))
ser.close()