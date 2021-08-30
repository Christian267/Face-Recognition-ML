import cv2 as cv
import math
import pathlib
import os

def face_scraper():
    """
    Crops all raw images in ./rawimages/test or ./rawimages/train into usable face images for classification.
    """
    base_directory = pathlib.Path(__file__).parent.absolute()
    test_or_train, is_target_face = ask_for_directory()
    folders = ['test', 'train']
    test_or_train = folders[test_or_train]
    source_directory = os.path.join(base_directory, 'rawimages', test_or_train, str(is_target_face))
    target_directory = os.path.join(base_directory, 'datasets', test_or_train, str(is_target_face))
    print('The source folder is ' + source_directory)
    print('The target folder is ' + target_directory)
    print('Files before saving images:')
    print(os.listdir(target_directory))
    crop_and_save_images(source_directory, target_directory)
    print('Files after saving images:')
    print(os.listdir(target_directory))

def crop_face_image(imageDirectory, haar_cascade, size=160):
    """
    Returns a square image of detected face cropped out of the given image, returns None if no face is detected
    :Param imageDirectory: str 
    :param haar_cascade: str 
    :param size: int
    :return: list
    """
    img = cv.imread(imageDirectory)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
    xtemp, ytemp, wtemp, htemp = 0, 0, 0, 0
    face_detected = False
    for(x,y,w,h) in faces_rect:
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
        if w < 500 or h <500:
            continue
        xtemp, ytemp, wtemp, htemp = x,y,w,h
        face_detected = True

    if not face_detected:
        return None
        
    x,y,w,h = xtemp, ytemp, wtemp, htemp
    crop = img[y:y + h, x:x + w]
    crop = cv.resize(crop, (size, size))
    return crop

def ask_for_directory():
    """
    Asks user to enter integer representing whether they are choosing to process their own images or images of others. Returns 0 for others,
    1 for self.
    :return: tuple(int, int)
    """
    while True:
        try:
            train_or_test = int(input('Are these images for training or testing? (0 = testing, 1 = training): '))
            target_or_random = int(input('Are these images of the person you want to identify? (0 = no, 1 = yes): '))
            
            if train_or_test in [0, 1] and target_or_random in [0, 1]:
                break
        except ValueError as e:
            print(f'{e}, Please enter proper values!')
        
    return (train_or_test, target_or_random)

def crop_and_save_images(source_directory, target_directory):
    """
    Goes through every image in the source_directory and crops the detected face from the image. 
    These cropped images are then all saved to the target_directory
    :param source_directory: str 
    :param target_directory: str
    return: None
    """
    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    os.chdir(target_directory)
    file_names = os.listdir(source_directory)
    number_of_images = len(file_names)
    total_number_of_digits = math.floor(math.log10(number_of_images)) + 1
    i = 1
    for filename in file_names:
        image = os.path.join(source_directory, filename)
        print(image)
        cropped_image = crop_face_image(image, haar_cascade)
        try:
            if cropped_image.any() is None:
                print('No face detected, skipping image')
                continue
        except AttributeError:
                print('No face detected, skipping image')
                continue
        current_number_of_digits = math.floor(math.log10(i)) + 1
        number_of_zeros = total_number_of_digits - current_number_of_digits
        print(f'Cropping: {filename}, saving to folder {target_directory}')
        newFilename = 'IMG_' + '0'*number_of_zeros + f'{i}.jpg'
        cv.imwrite(newFilename, cropped_image)
        i += 1

def renameImages(directory):
    """
    Renames all .jpg or .png in a given directory into standardized names (e.g. IMG_001.jpg). Not 
    necessary but helps with data cleaning by expediting the search for corresponding images of poorly cropped face images. 
    :param directory: str
    """
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