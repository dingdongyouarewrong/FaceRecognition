
from keras import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, MaxPool2D, Flatten, Dropout, np, Activation
from keras_applications.densenet import preprocess_input
from keras_preprocessing.image import load_img, img_to_array

# distance between faces
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

# distance between faces
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# img2array
def preprocessing_image(image_path):
    image = load_img(image_path, target_size=(260,270))
    image = img_to_array(image)
    #add another dimension
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image, data_format="channels_last")
    return image

def create_model():
    model = Sequential()
    model.add(ZeroPadding2D(input_shape=(224,224,3), data_format="channels_last"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    model.add(Conv2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(2622, (1, 1)))
    model.add(Flatten())

    model.add(Activation('softmax'))
    model.load_weights("weights/vgg_face_weights.h5")

# new network from first layer to previous of output layer
    vgg_face_descriptor = Model(inputs=model.layers[0].input
                                , outputs=model.layers[-2].output)
    return vgg_face_descriptor