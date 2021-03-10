import os
import math

myDir = os.getcwd()
targetDirectory = os.path.join(myDir, 'RawImageData', 'myFaces')
filenames = os.listdir(targetDirectory)
numberOfImages = len(filenames)
totalNumberOfDigits = math.floor(math.log10(numberOfImages)) + 1

i = 1
print('The current folder is ' + targetDirectory)
for filename in filenames:
    currentNumberOfDigits = math.floor(math.log10(i)) + 1
    numberOfZeros = totalNumberOfDigits - currentNumberOfDigits
    print(f'renaming: {filename} to' + 'IMG_' + '0'* numberOfZeros + f'{i}.jpg')
    oldFile = os.path.join(targetDirectory, filename)
    newFile = os.path.join(targetDirectory, 'IMG_' + '0'* numberOfZeros + f'{i}.jpg')
    os.rename(oldFile, newFile)
    i += 1

print()