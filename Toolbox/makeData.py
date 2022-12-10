from core.FaRL_faceparser import FaRL_256, FaRL_512, FaRL_1024
import glob
import os
import sys
import cv2
import numpy as np
import PIL
from shutil import copyfile
import random
from PIL import Image
from core.affine_512 import affine_img, affine_mask, warp_image, find_invH
import torch
from torchvision import transforms
from torchvision.utils import save_image



def divideTrainTest():
    path =  "./KID-F/test"
    dst = "./KID-F/PreAffine/"

    for filepath in glob.glob(path+'/*.jpg'):
            name = filepath.split("/")
            filepath2 = os.path.join(dst,name[-1])
            copyfile(filepath, filepath2) 

    path =  "./KID-F/train"
    for filepath in glob.glob(path+'/*.jpg'):
            name = filepath.split("/")
            filepath2 = os.path.join(dst,name[-1]) 
            copyfile(filepath, filepath2)


    path = "./KID-F/PreAffine"
    dst1 = "./KID-F/test_PreAffine/"
    dst2 = "./KID-F/train_PreAffine/"
    filelist = glob.glob(path+'/*.jpg')
    testlist = random.sample([i for i in range(len(filelist))], 300)
    for i in range(len(filelist)):
        filepath = filelist[i]
        name = filepath.split("/")
        if (i in testlist):
            filepath2 = os.path.join(dst1,name[-1])
        else:
            filepath2 = os.path.join(dst2,name[-1])
        copyfile(filepath, filepath2)

def makeMaskData():
    path = "./KID-F/test_PreAffine"
    affined_dst = "./KID-F/test_Affined/"
    mask_dst = "./KID-F/test_Mask/"
    for filepath in glob.glob(path+'/*.jpg'):
        idx = filepath.split("/")[-1]
        print(idx[:-4], end="")
        try:
            mat_img = Image.open(filepath).resize((512, 512))
            mask_pred_512 = FaRL_512(mat_img)
            tensor_image_512 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                            transforms.ToTensor()(mat_img.resize((512,512))))
            affined_img_tensor = affine_img(tensor_image_512, mask_pred_512)
            affined_mask_tensor = affine_mask(mask_pred_512)
            affined_img = transforms.ToPILImage()((affined_img_tensor+1)/2)
            affined_img.save(affined_dst+idx)
            affined_mask_tensor[0,2]=affined_mask_tensor[0,2]+affined_mask_tensor[0,1]
            affined_mask_tensor[0,1]=affined_mask_tensor[0,0]
            idx = idx[:-4]
            for i in range(4):
                affined_mask = transforms.ToPILImage()(affined_mask_tensor[0,i+1])
                if i==0:
                    affined_mask.save(mask_dst+idx+"_skin.jpg")
                elif i==1:
                    affined_mask.save(mask_dst+idx+"_eye.jpg")
                elif i==2:
                    affined_mask.save(mask_dst+idx+"_nose.jpg")
                else:
                    affined_mask.save(mask_dst+idx+"_mouth.jpg")
            print('\t'+"done")
        except np.linalg.LinAlgError:
            print('\t'+"failed")
            pass


if __name__ == '__main__':
    # divideTrainTest()
    makeMaskData()
    


