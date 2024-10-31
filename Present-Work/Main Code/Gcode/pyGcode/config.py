class config():
    def __init__(self,):
        
        # Numbers found by iterative process.
        # Refer CAD file of finger mount.

        # NOTE : Shift in x is remaining
        self.xstart = 99.7 #92
        self.ystart = 57.1 #66
        self.zstart = 10.5

        self.z_elevate = 5
        self.xtravel = 34 #in mm
        self.ytravel = 108 #in mm
        self.ztravel = 0  # in mm
        self.xshift = 2.5
        self.feed = 6000 #mm/min z------> 5 mm/sec(Max out)
        
        self.gcodefile = "fingergcode.txt"
        with open('./pyGcode/files/startgcode.txt') as f:
            self.start_text = f.readlines()
        
        with open('./pyGcode/files/endgcode.txt') as f:
            self.end_text = f.readlines()
