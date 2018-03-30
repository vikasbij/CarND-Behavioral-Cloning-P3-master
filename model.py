import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
import sklearn
from sklearn.utils import shuffle
from keras.layers import Flatten,Dense,Dropout
samples = []
with open('../driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader,None)
    for line in reader:
        samples.append(line)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def preprocess(image):
    
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    normalized = image/255.0 - 0.5
    resized = cv2.resize(normalized[50:140,:],(64,64))
    #normalized = resized/255.0 - 0.5
    return resized

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images=[]
            angles=[]
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                centre_name = '../IMG/'+batch_sample[0].strip().split('/')[-1]
                center_image = cv2.imread(centre_name)
                if center_image is None:
                    print("invalid image path:", centre_name)
                center_image_processed =preprocess(center_image)
                center_image_flipped=cv2.flip(center_image_processed,1)
                left_name = '../IMG/'+batch_sample[1].strip().split('/')[-1]
                left_image = cv2.imread(left_name)
                if left_image is None:
                    print("invalid image path:", left_name)
                    
                left_image_processed =preprocess(left_image)
                left_image_flipped=cv2.flip(left_image_processed,1)
                right_name = '../IMG/'+batch_sample[2].strip().split('/')[-1]
                right_image = cv2.imread(right_name)
                if right_image is None:
                    print("invalid image path:", right_name)
                right_image_processed =preprocess(right_image)
                right_image_flipped=cv2.flip(right_image_processed,1)
                left_angle = center_angle + 0.20
                right_angle = center_angle - 0.23
                images.extend([center_image_processed, center_image_flipped, left_image_processed,right_image_processed,left_image_flipped,right_image_flipped])
                
                angles.extend([center_angle, -1*center_angle, left_angle,right_angle,-left_angle,-right_angle])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

train_generator = generator(train_samples, batch_size=8)

# for i in range(10):
#     x, y = next(train_generator)
#     print(x.shape, y.shape)

#rint(next(train_generator))
validation_generator = generator(validation_samples, batch_size=8)
model=Sequential()
model.add(Convolution2D(16, 5, 5, activation='relu', subsample=(2, 2), input_shape=(64,64,3)))
model.add(Convolution2D(32, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(32, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Dropout(1.0))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6,validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=2)
model.save('model.h5')
