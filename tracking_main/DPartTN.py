

#image =Image.open('00000002.jpg').convert('RGB')
#selection = np.array([130.,115.,85.,99.])
#tracker = DPartTN_tracker(image, selection)
#region = tracker.track(image)
#region = vot.Rectangle(region[0],region[1],region[2],region[3])



import vot
import sys
import cv2

reload(sys)
sys.setdefaultencoding('utf8')
import PIL.Image as Image
import numpy as np
from DPartTN_Tracker import *
handle = vot.VOT("rectangle")
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

selection = handle.region()
selection = np.array([selection.x,selection.y,selection.width,selection.height])
image =Image.open(imagefile).convert('RGB')
tracker = DPartTN_tracker(image, selection)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = Image.open(imagefile).convert('RGB')
    region= tracker.track(image)
    region = vot.Rectangle(region[0],region[1],region[2],region[3])
    handle.report(region, 1.0000000)