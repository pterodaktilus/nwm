
from PIL import Image,ImageFilter
import cupy 
import os
import math as m
import cv2 
import numpy as np

pool_avg = cupy.ElementwiseKernel(
    'T x,T y,T z,T w,',
    "T a",
    "a = (x + y + z + w)/4",
    "pool_avg")

def load2memory(img_path):
    test = Image.open(img_path)
    resized = test.resize((200,200))
    #resized = cupy.asarray(resized)
    return resized


def sigmoid(val): # zamenit za ReLu ?
    return 1/ (1 + m.e ** (-val))

def convolve(image):  
    #image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=250, threshold=3))
    #image.show()
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image, 100, 200)
    return edges

def pool(img): #https://en.wikipedia.org/wiki/Convolutional_neural_network#Architecture
    pooled = cupy.zeros((len(img)/2,len(img[0])/2,3),3,dtype=cupy.uint8)
    for i in range(len(img),2):
        for j in range(len(img[0]),2):
            pooled[i][j] = pool_avg(img[i][j],img[i+1][j],img[i][j+1],img[i+1][j+1])
            
    
    
def train(path):
    length = len(os.listdir(path))
    for i in range(1,length+1):
        img_path = path + f"\\cat.{i}.jpg"
        img = (load2memory(img_path))
        convolve(img)
        img = pool(img)
        Image.fromarray(img).show()
        break
        
        
        
def main():
    path = os.getcwd()+"\\dataset\\training_set\\cats"
    train(path)

main()
