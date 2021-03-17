import os
import math

def renameImages(directory):
    '''
    Renames all .jpg or .png in a given directory into standardized names (e.g. IMG_001.jpg). Not 
    necessary but helps with data cleaning by expediting the search for corresponding images of poorly cropped face images. 
    '''


    filenames = os.listdir(directory)
    numberOfImages = len(filenames)
    totalNumberOfDigits = math.floor(math.log10(numberOfImages)) + 1

    i = 1
    print('The current folder is ' + directory)
    for filename in filenames:
        currentNumberOfDigits = math.floor(math.log10(i)) + 1
        numberOfZeros = totalNumberOfDigits - currentNumberOfDigits
        print(f'renaming: {filename} to' + 'IMG_' + '0'* numberOfZeros + f'{i}.jpg')
        oldFile = os.path.join(directory, filename)
        newFile = os.path.join(directory, 'IMG_' + '0'* numberOfZeros + f'{i}.jpg')
        os.rename(oldFile, newFile)
        i += 1

    print()


myDir = os.getcwd()
targetDirectory = os.path.join(myDir, 'rawimages', 'train', '1')
renameImages(targetDirectory)