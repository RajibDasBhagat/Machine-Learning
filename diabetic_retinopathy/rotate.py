#Do this before preprocessing
from scipy import ndimage, misc
import numpy as np
import os
import cv2

def main(degree,inpath,outPath):

    print("rotating....!!")
    # iterate through the names of contents of the folder
    for image_path in os.listdir(inpath):

        # create the full input path and read the file
        input_path = os.path.join(inpath, image_path)
        image_to_rotate = ndimage.imread(input_path)

        # rotate the image
        rotated = ndimage.rotate(image_to_rotate, degree)

        # create full output path, 'example.jpg' 
        # becomes 'rotate_example.jpg', save the file to disk
        fullpath = os.path.join(outPath, 'rot'+str(degree)+'_'+image_path)
        misc.imsave(fullpath, rotated)
    print(str(degree) +" Completed...!!")
if __name__ == '__main__':
    #outpath1 = "G://mtp3/2.group/train/x1/"
    #inpath1 = "G://mtp3/2.group/train/1/"
    #outpath2 = "G://mtp3/2.group/train/x2/"
    #inpath2 = "G://mtp3/2.group/train/2/"
    #outpath3 = "G://mtp3/2.group/train/x3/"
    #inpath3 = "G://mtp3/2.group/train/3/"
    outpath4 = "G://mtp3/2.group/train/x4/"
    inpath4 = "G://mtp3/2.group/train/4/"
    
    #count1=10
    #count2=4
    #count3=29
    count4=5
    #for i in range(count1):
    #    main(i+5,inpath1,outpath1)
    #for i in range(count2):
    #    main(i+10,inpath2,outpath2)
    #for i in range(count3):
    #    main(i+3,inpath3,outpath3)
    for i in range(count4):
        main((i+1)*10,inpath4,outpath4) 