
import cv2, os, glob
import numpy as np
print("ok1")


# In[2]:


scale=300   #change value as required
test_src0="/scratch/scratch2/_retinopathy/2.group_test/0/*.*"
test_src1="/scratch/scratch2/_retinopathy/2.group_test/1/*.*"
test_src2="/scratch/scratch2/_retinopathy/2.group_test/2/*.*"
test_src3="/scratch/scratch2/_retinopathy/2.group_test/3/*.*"
test_src4="/scratch/scratch2/_retinopathy/2.group_test/4/*.*"


test_dest0="/scratch/scratch2/_retinopathy/4.green_test/0/"
test_dest1="/scratch/scratch2/_retinopathy/4.green_test/1/"
test_dest2="/scratch/scratch2/_retinopathy/4.green_test/2/"
test_dest3="/scratch/scratch2/_retinopathy/4.green_test/3/"
test_dest4="/scratch/scratch2/_retinopathy/4.green_test/4/"

train_src0="/scratch/scratch2/_retinopathy/2.group_train/0/*.*"
train_src1="/scratch/scratch2/_retinopathy/2.group_train/1/*.*"
train_src2="/scratch/scratch2/_retinopathy/2.group_train/2/*.*"
train_src3="/scratch/scratch2/_retinopathy/2.group_train/3/*.*"
train_src4="/scratch/scratch2/_retinopathy/2.group_train/4/*.*"

train_dest0="/scratch/scratch2/_retinopathy/4.green_train/0/"
train_dest1="/scratch/scratch2/_retinopathy/4.green_train/1/"
train_dest2="/scratch/scratch2/_retinopathy/4.green_train/2/"
train_dest3="/scratch/scratch2/_retinopathy/4.green_train/3/"
train_dest4="/scratch/scratch2/_retinopathy/4.green_train/4/"

def make_dir(dest):
    if not os.path.exists(dest):
        os.mkdir(dest)

make_dir(test_dest0)
make_dir(test_dest1)
make_dir(test_dest2)
make_dir(test_dest3)
make_dir(test_dest4)

make_dir(train_dest0)
make_dir(train_dest1)
make_dir(train_dest2)
make_dir(train_dest3)
make_dir(train_dest4)

print("ok2")


# In[3]:


def scaleRadius(img,scale):
    x=img[img.shape[0]/2,:,:].sum(1)
    radius=(x>x.mean()/10).sum()/2
    s=scale*1.0/radius
    return cv2.resize(img,(512,512),fx=s,fy=s)


def pre_processing(src, dest, flag):
    for file in glob.glob(src):
        try:
             	#input image
        	img1=cv2.imread(file,1)           #flag 1,0,-1 color/gray/unchanged
        	#cv2.imshow("image",img1)

       		#scale
        	img2=scaleRadius(img1,scale)

        	green_channel = img2[:, :, 1]

        	#Apply Clahe twice
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cla = clahe.apply(green_channel)
		img3 = clahe.apply(cla)

        	if(flag==True):
            		cv2.imwrite(str(dest) + file[46:],img3)
        	else:
            		cv2.imwrite(str(dest) + file[47:],img3)

        	#cv2.destroyAllWindows()
        except:
                print("Error")



print(" __________Green Cliche Pre-processing__________")

flag=True
print("Working folder test0")
pre_processing(test_src0, test_dest0, flag)
print("Working folder test1")
pre_processing(test_src1, test_dest1, flag)
print("Working folder test2")
pre_processing(test_src2, test_dest2, flag)
print("Working folder test3")
pre_processing(test_src3, test_dest3, flag)
print("Working folder test4")
pre_processing(test_src4, test_dest4, flag)

flag=False
print("Working folder train0")
pre_processing(train_src0, train_dest0, flag)
print("Working folder train1")
pre_processing(train_src1, train_dest1, flag)
print("Working folder train2")
pre_processing(train_src2, train_dest2, flag)
print("Working folder train3")
pre_processing(train_src3, train_dest3, flag)
print("Working folder train4")
pre_processing(train_src4, train_dest4, flag)

print("Pre-process Green clache completed")
