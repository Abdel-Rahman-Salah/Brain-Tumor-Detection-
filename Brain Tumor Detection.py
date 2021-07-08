# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 22:02:50 2021

@author: Abdelrahman Salah 
"""

import cv2

import numpy as np
import os
import matplotlib.pyplot as plt
import math
import numpy
import glob

def train(path):
    sumcorrel=0
    count=0
    for x in glob.glob(path):
        img = cv2.imread(x)
        img=cv2.imread(x)
        height = img.shape[0]
        width=img.shape[1]
        width_cutoff = width // 2
        s1 = img[:, :width_cutoff]
        s2 = img[:, width_cutoff:] 
        hist1 = cv2.calcHist([s1],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([s2 ],[0],None,[256],[0,256])
        #plt.hist(s1.ravel(),256,[0,256]); plt.show()
        #plt.hist(s2 .ravel(),256,[0,256]); plt.show()
        correl=cv2.compareHist(hist1, hist2,  cv2.HISTCMP_CORREL )
        count=count+1
        sumcorrel=sumcorrel+correl 
    avgcorrel=sumcorrel/count
    return avgcorrel,count;

def test(path,avgyesno):
        for x in glob.glob(path):
            img = cv2.imread(x)
            img=cv2.imread(x)
            height = img.shape[0]
            width=img.shape[1]
            # Cut the image in half
            width_cutoff = width // 2
            s1 = img[:, :width_cutoff]
            s2 = img[:, width_cutoff:]
            cv2.imshow("s1",s1)
            cv2.imshow("s2",s2)
            cv2.waitKey(0)

            cv2.destroyAllWindows()
            hist1 = cv2.calcHist([s1],[0],None,[256],[0,256])
            hist2 = cv2.calcHist([s2 ],[0],None,[256],[0,256])
            plt.hist(s1.ravel(),256,[0,256]); plt.show()
            plt.hist(s2 .ravel(),256,[0,256]); plt.show()
            correl=cv2.compareHist(hist1, hist2,  cv2.HISTCMP_CORREL)
            print("Loaded a test image ")
            print(x)
            print("Correlation:")
            print(correl)
            if(correl>avgyesno):
             print("Classification : No ")
            else:
             print("Classification : Yes ")
           
           
path_yes_training=r'C:\Users\lenevo\Desktop\Spring 2021\Image processing\project\Dataset\Training\yes\*.jpg'
path_no_training=r'C:\Users\lenevo\Desktop\Spring 2021\Image processing\project\Dataset\Training\no\*.jpg'
path_yes_test=r'C:\Users\lenevo\Desktop\Spring 2021\Image processing\project\Dataset\Testing\yes\*.jpg'
path_no_test=r'C:\Users\lenevo\Desktop\Spring 2021\Image processing\project\Dataset\Testing\no\*.jpg'

avgcorrel1,count=train(path_yes_training)
avgcorrel2,count2=train(path_no_training)


print("Training completed using a dataset of Brain MRI Images..\n")
print("Average Correlation for Yes class (tumor exist)  : \n") 
print(avgcorrel1 )  
print("Average Correlation for No class (no tumor)  :  \n") 
print(avgcorrel2)
test(path_yes_test,(avgcorrel1+avgcorrel2)/2)
test(path_no_test,(avgcorrel1+avgcorrel2)/2)





