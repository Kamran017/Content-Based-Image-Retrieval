'''
    17011904 KAMRAN BALAYEV
'''

import cv2
import math
import numpy as np
import glob

# min max normalization function
def Normalization(maxNum, minNum, arr):
    arr2 = np.zeros((len(arr)), dtype=np.float32)
    for i in range(len(arr)):
        arr2[i] = (arr[i] - minNum) / (maxNum - minNum)
    return arr2

#calculate ecludian distance
def Euclidian(arr1, arr2):
    eucArr = 0.0
    for i in range(len(arr1)):
        eucArr = eucArr + pow((arr1[i] - arr2[i]), 2)
    return math.sqrt(eucArr)


#create histogram
def createHistogram(arr):
    hist = np.zeros((256), dtype=np.float32)
    for i in arr:
        hist[i] = hist[i] + 1
    return hist


def createHueHistogram(img1):
    hsvImage = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    Hue_Hist = np.zeros((256), dtype=np.float32)
    for i in hsvImage[:, :, 0]:
        Hue_Hist[i] = Hue_Hist[i] + 1
    maxH = max(Hue_Hist)
    minH = min(Hue_Hist)
    normalHueHistogram = Normalization(maxH, minH, Hue_Hist)
    return normalHueHistogram

def calculateImgEuc(normalHistogram, normalHistogram2):
    imgEuc = Euclidian(normalHistogram, normalHistogram2)
    return imgEuc


def createRGBhistogram(img):
    histogramList = []
    # Red
    red = img[:, :, 0]
    redHistogram = createHistogram(red)
    # Red Normalization
    maximumRed = max(redHistogram)
    minimumRed = min(redHistogram)
    normalRedHistogram = Normalization(maximumRed, minimumRed, redHistogram)
    # Green
    green = img[:, :, 1]
    greenHistogram = createHistogram(green)
    # Green Normalization
    maximumGreen = max(greenHistogram)
    minimumGreen = min(greenHistogram)
    normalGreenHistogram = Normalization(maximumGreen, minimumGreen, greenHistogram)
    # Blue
    blue = img[:, :, 2]
    blueHistogram = createHistogram(blue)
    # Blue Normalization
    maximumBlue = max(blueHistogram)
    minimumBlue = min(blueHistogram)
    normalBlueHistogram = Normalization(maximumBlue, minimumBlue, blueHistogram)
    # Adding to list
    histogramList.append(normalRedHistogram)
    histogramList.append(normalGreenHistogram)
    histogramList.append(normalBlueHistogram)
    return histogramList


def rbgEucCalc(img, img2):
    # R G B Histograms
    rbgHistLists = createRGBhistogram(img)
    rgbHistLists2 = createRGBhistogram(img2)
    rgbEucList = []
    # Calculate distances then append them to the list and return it
    R = calculateImgEuc(rbgHistLists[0], rgbHistLists2[0])
    G = calculateImgEuc(rbgHistLists[1], rgbHistLists2[1])
    B = calculateImgEuc(rbgHistLists[2], rgbHistLists2[2])
    rgbEucList.append(R)
    rgbEucList.append(G)
    rgbEucList.append(B)
    return rgbEucList

#After calculating euc distances between train and test images find the minimum 5 values with this function
def findMinVals(tmp,minimumTmp,element):
    # define minimum value minimumTmp with high value in order to reach the minimum values in list
    finalList = []
    flag = []
    # find minimum values remove them from tmp list then find other values repeat 5 times
    for x in range(5):
        for y in tmp:
            if (y[element] < minimumTmp):
                minimumTmp = y[element]
                flag = y.copy()
                tmp.remove(y)
        finalList.append(flag)
        minimumTmp = 10000
    return finalList

pngfiles = [] #store train files
pngFilesTest = [] #store test files

for file in glob.glob("train/*.jpg"):
    pngfiles.append(file)

for file in glob.glob("test/*.jpg"):
    pngFilesTest.append(file)

dictionaryList = []
imageDictionary = {
    "Image1": "img1",
    "Image2": "img2",
    "R_Dist": 0,
    "G_Dist": 0,
    "B_Dist": 0,
    "H_Dist": 0
}


for i in range(len(pngfiles) - 1):
    img1 = cv2.imread(pngfiles[i])
    img1_hue_hist = createHueHistogram(img1)
    #read test photo from folder
    for a in range(i, len(pngFilesTest)):
        # read test photo
        img2 = cv2.imread(pngFilesTest[a])
        img2_hue_hist = createHueHistogram(img2) #create hue histogram of photo
        hueEucDist = calculateImgEuc(img1_hue_hist, img2_hue_hist)
        rgbList = rbgEucCalc(img1, img2)
        imageDictionary["Image1"] = pngfiles[i]
        imageDictionary["Image2"] = pngFilesTest[a]
        imageDictionary["R_Dist"] = rgbList[0]
        imageDictionary["G_Dist"] = rgbList[1]
        imageDictionary["B_Dist"] = rgbList[2]
        imageDictionary["H_Dist"] = hueEucDist
        dictionaryList.append(imageDictionary.copy())

# CREATE LIST FOR R
RfinalList=findMinVals(dictionaryList.copy(), 5000, "R_Dist")

# CREATE LIST FOR G
GfinalList = findMinVals(dictionaryList.copy(), 5000, "G_Dist")

# CREATE LIST FOR B

BfinalList = findMinVals(dictionaryList.copy(), 5000, "B_Dist")

# CREATE LIST FOR H
HfinalList = findMinVals(dictionaryList.copy(), 5000, "H_Dist")

print("R")
for list in RfinalList:
    print(list)
    print('\n')

print("G")
for list in GfinalList:
    print(list)
    print('\n')

print("B")
for list in BfinalList:
    print(list)
    print('\n')

print("H")
for list in HfinalList:
    print(list)
    print('\n')

