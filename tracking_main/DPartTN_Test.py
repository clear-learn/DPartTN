import vot
import sys
import cv2

reload(sys)
sys.setdefaultencoding('utf8')
import PIL.Image as Image
import numpy as np
from DPartTN_Tracker import *
path = './'

name = '0000000'
image =Image.open(path+'00000001.jpg').convert('RGB')
#selection = np.array([891.0,330.0,914.0,330.0,914.0,337.0,891.0,337.0])
selection = np.array([128.7200,458.3600,28.1100,71.0500])
#selection = np.array([891.0000,330.0000,23.0000,17.0000])
tracker = DPartTN_tracker(image, selection)

for i in range(2,9):

    if(i==2):
        pass
    image = Image.open(path + '0000000'+str(i)+'.jpg').convert('RGB')
    region = tracker.track(image)
    print(region)


region = vot.Rectangle(region[0],region[1],region[2],region[3])
