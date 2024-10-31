class config():
    def __init__(self,):
        self.xstart = 80
        self.ystart = 80
        self.zstart = 20
        self.xtravel = 36 #in mm
        self.ytravel = 108 #in mm
        self.xshift = 1
        self.ztravel = 0  # in mm
        self.feed = 1200 #inch/min
        
        self.gcodefile = "output/fingergcode.txt"
        with open('files/startgcode.txt') as f:
            self.start_text = f.readlines()
        
        with open('files/endgcode.txt') as f:
            self.end_text = f.readlines()
