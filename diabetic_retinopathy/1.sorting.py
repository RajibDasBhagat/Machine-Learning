import shutil
import csv
import os

#array list to store the selected file names
#global declaration
arr_normal    = []
arr_mild      = []
arr_moderate  = []
arr_severe    = []
arr_proliferative = []

src_train   =  "/scratch/scratch2/_retinopathy/1.train/"
src_test    =  "/scratch/scratch2/_retinopathy/1.test/" 



trainLabels = '/scratch/scratch2/_retinopathy/trainLabels.csv'    
testLabels  = '/scratch/scratch2/_retinopathy/testLabels.csv'    

def counting(file_path):
    count_rows = 0  #count the total number of rows i.e training files

    #category-wise counting as per dataset label
    #countnormal=count mild nonproliferative=count moderate nonproliferative=count severe nonproliferative=count proliferative
    count_0 = count_1 = count_2 = count_3 = count_4 = 0

    with open(file_path, 'r') as csvfile:
        f = csv.reader(csvfile)
        for row in f:
            count_rows += 1
            #print(row[0])
            c=(' '.join(row[1]))   #select column containing values such as 0, 1, 2, 3, 4
            #print(v)
            if(c == str(0)):
                count_0 += 1
                arr_normal.append(row[0]+'.jpeg')
            elif(c == str(1)):
                count_1 += 1
                arr_mild.append(row[0]+'.jpeg')
            elif(c == str(2)):
                count_2 += 1
                arr_moderate.append(row[0]+'.jpeg')
            elif(c == str(3)):
                count_3 += 1
                arr_severe.append(row[0]+'.jpeg')
            elif(c == str(4)):
                count_4 += 1
                arr_proliferative.append(row[0]+'.jpeg')

        total = count_0 + count_1 + count_2 + count_3 + count_4
        #print(count_0, count_1, count_2, count_3, count_4 )
        print(len(arr_normal), len(arr_mild), len(arr_moderate), len(arr_severe),len(arr_proliferative))
    return count_0, count_1, count_2, count_3, count_4

def percentage(count_0, count_1, count_2, count_3, count_4):
    percent = 100    #change value as required
    
    #percentage split
    percent_normal   = int(count_0 * percent / 100)
    percent_mild     = int(count_1 * percent / 100)
    percent_moderate = int(count_2 * percent / 100)
    percent_severe   = int(count_3 * percent / 100)
    percent_proliferative = int(count_4 * percent / 100)
    print(percent_normal, percent_mild, percent_moderate, percent_severe, percent_proliferative)
    #for i in arr_normal: print (i)
    count_percent = percent_normal + percent_mild + percent_moderate + percent_severe + percent_proliferative
    #print(count_percent)
    return percent_normal, percent_mild, percent_moderate, percent_severe, percent_proliferative

def make_directory(dest):
    if not os.path.exists(dest):
        os.mkdir(dest)	

def sorting_images(src,dest0,dest1,dest2,dest3,dest4,count_0,count_1,count_2,count_3,count_4):
    count_normal = count_mild = count_severe = count_moderate = count_proliferative = 0
    print("Please wait sorting images to folders...!!")
    for file_name in os.listdir(src):
        #print(file_name)
        if(file_name in arr_normal and count_normal < count_0):
            full_file_name = os.path.join(src, file_name)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, dest0)
                count_normal += 1

        if(file_name in arr_mild and count_mild < count_1):
            full_file_name = os.path.join(src, file_name)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, dest1)
                count_mild += 1

        if(file_name in arr_moderate and count_moderate < count_2): 
            full_file_name = os.path.join(src, file_name)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, dest2)
                count_moderate += 1

        if(file_name in arr_severe and count_severe < count_3):
            full_file_name = os.path.join(src, file_name)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, dest3)
                count_severe += 1

        if(file_name in arr_proliferative and count_proliferative < count_4): 
            full_file_name = os.path.join(src, file_name)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, dest4)
                count_proliferative += 1
    print("Images sorted...!!")
    print("Total files sorted:  "+ str(count_normal+count_mild+count_severe+count_moderate+count_proliferative))


def clear_array(arr_normal, arr_mild, arr_moderate, arr_severe, arr_proliferative):
    arr_normal    = []
    arr_mild      = []
    arr_moderate  = []
    arr_severe    = []
    arr_proliferative = [] 
    return arr_normal, arr_mild, arr_moderate, arr_severe, arr_proliferative





#counting train images
print("               _____Train______              ")
count_0, count_1, count_2, count_3, count_4 = counting(trainLabels)
#percent_normal, percent_mild, percent_moderate, percent_severe, percent_proliferative = percentage(count_0, count_1, count_2, count_3, count_4)

#make directories for group
dest_train0   = "/scratch/scratch2/_retinopathy/2.group_train/0/"    
dest_train1   = "/scratch/scratch2/_retinopathy/2.group_train/1/"
dest_train2   = "/scratch/scratch2/_retinopathy/2.group_train/2/"
dest_train3   = "/scratch/scratch2/_retinopathy/2.group_train/3/"
dest_train4   = "/scratch/scratch2/_retinopathy/2.group_train/4/"

make_directory(dest_train0)
make_directory(dest_train1)
make_directory(dest_train2)
make_directory(dest_train3)
make_directory(dest_train4)



print("         ______Sorting train images!!______    ")
sorting_images(src_train, dest_train0, dest_train1, dest_train2, dest_train3, dest_train4,count_0,count_1,count_2,count_3,count_4)
print("Completed!!")


#delete the previous value from the array list      
arr_normal, arr_mild, arr_moderate, arr_severe, arr_proliferative = clear_array(arr_normal, arr_mild, arr_moderate, arr_severe, arr_proliferative)

#counting test images
print("                _____Test_____               ")
count_0, count_1, count_2, count_3, count_4 = counting(testLabels)
#percent_normal, percent_mild, percent_moderate, percent_severe, percent_proliferative = percentage(count_0, count_1, count_2, count_3, count_4)


dest_test0   =  "/scratch/scratch2/_retinopathy/2.group_test/0/"    
dest_test1   =  "/scratch/scratch2/_retinopathy/2.group_test/1/"
dest_test2   =  "/scratch/scratch2/_retinopathy/2.group_test/2/"
dest_test3   =  "/scratch/scratch2/_retinopathy/2.group_test/3/"
dest_test4   =  "/scratch/scratch2/_retinopathy/2.group_test/4/"

#make directories for group
make_directory(dest_test0)
make_directory(dest_test1)
make_directory(dest_test2)
make_directory(dest_test3)
make_directory(dest_test4)

#sort images according to group
print("          _____Sorting test images!!______     ")
sorting_images(src_test, dest_test0, dest_test1, dest_test2, dest_test3, dest_test4,count_0,count_1,count_2,count_3,count_4)
print(" sorting Completed!!")