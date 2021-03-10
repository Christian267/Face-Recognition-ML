import cv2 as cv
import os
import math

def cropFaceImage(imageDirectory, size=160):
    '''
    Returns a square image of detected face cropped out of the given image, returns None if no face is detected
    Param: imageDirectory: the directory of target image
                     size: pixel size of returned image
    '''
    
    img = cv.imread(imageDirectory)
    # cv.imshow('faceImage', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray', gray)

    haar_cascade = cv.CascadeClassifier(r'C:\Users\14694\MPS\tf-venv\Facial Recognition CNN\FaceDataProcessing\haar_face.xml')
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
    
    xtemp, ytemp, wtemp, htemp = 0, 0, 0, 0
    facePresent = False     # Flag for determining if a face was detected
    for(x,y,w,h) in faces_rect:
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
        if w < 500 or h <500:
            continue
        xtemp, ytemp, wtemp, htemp = x,y,w,h
        facePresent = True

    if not facePresent:
        return None
        
    x,y,w,h = xtemp, ytemp, wtemp, htemp
    crop = gray[y:y + h, x:x + w]
    crop = cv.resize(crop, (size, size))
    return crop
    # cv.imshow('Cropped Face', crop)
    # cv.imshow('Detected Faces', img)
    # cv.waitKey(0)

def askWhoseImages():
    '''
    Asks user to enter integer representing whether they are choosing to process their own images or images of others. Returns 0 for others, 1 for self.
    '''
    while True:
        try:
            yourImages = int(input('Enter "1" if processing your images in the myfaces directory. Enter "0" if processing others in the "otherfaces" directory (1/0): '))
            if yourImages in [0,1]:
                break
        except ValueError as e:
            print(f'{e}, Please enter the integer values "0" or "1".')
        
    return yourImages

def cropAndSaveImages(sourceDirectory, targetDirectory):
    '''Goes through every image in the sourceDirectory and crops the detected face from the image. 
       These cropped images are then all saved to the targetDirectory
       Params: sourceDirectory: directory containing the images to be processed
               targetDirectory: directory to save the new cropped images
       
       '''

    os.chdir(targetDirectory)
    filenames = os.listdir(sourceDirectory)
    numberOfImages = len(filenames)
    totalNumberOfDigits = math.floor(math.log10(numberOfImages)) + 1
    i = 1
    for filename in filenames:
        image = os.path.join(sourceDirectory, filename)
        print(image)
        croppedImage = cropFaceImage(image)
        try:
            if not croppedImage.any():
                print('No face detected, skipping image')
                continue
        except AttributeError:
                print('No face detected, skipping image')
                continue
        currentNumberOfDigits = math.floor(math.log10(i)) + 1
        numberOfZeros = totalNumberOfDigits - currentNumberOfDigits
        image_path = os.path.join(sourceDirectory,'rawimages', 'myfacescropped', 'IMG_' + '0'*numberOfZeros + f'{i}.jpg')
        print(f'Cropping: {filename}, saving to folder {targetDirectory}')
        newFilename = 'IMG_' + '0'*numberOfZeros + f'{i}.jpg'
        cv.imwrite(newFilename, croppedImage)
        i += 1



if __name__ =='__main__':
    myDir = os.getcwd()
    sourceDirectory = [os.path.join(myDir, 'rawimages', 'otherfaces') ,os.path.join(myDir, 'rawimages', 'myfaces')]
    targetDirectory = [os.path.join(myDir, 'rawimages', 'otherfacescropped'), os.path.join(myDir, 'rawimages', 'myfacescropped')]

    yourImages = askWhoseImages()
    sourceFolder = sourceDirectory[yourImages]
    targetFolder = targetDirectory[yourImages]

    print('The target folder is ' + targetFolder)
    print('before saving images: ')
    print(os.listdir(targetFolder))

    cropAndSaveImages(sourceFolder, targetFolder)

    print('After saving images:')
    print(os.listdir(targetFolder))