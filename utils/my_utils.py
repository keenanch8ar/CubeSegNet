import cv2
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import segmentation_models as sm
from keras.utils import to_categorical

def merge_classes(mask, cls1, cls2, raw_num_classes):
    mask[mask == cls1] = cls2
    for i in range(cls1 + 1, raw_num_classes):
        mask[mask==i] = i - 1
    return mask


def show_mask_with_label(batch, labels, _min, _max, alpha=1,legend=True, ticks=True):
    values, counts = np.unique(batch, return_counts=True)
    counts = counts / (64*64)
    im = plt.imshow(batch, cmap='viridis', vmin = _min, vmax = _max, alpha=alpha)
    if ticks == False:
        plt.xticks([])
        plt.yticks([])
    colors = [im.cmap(im.norm(value)) for value in values]
    if legend == True:
        patches = [mpatches.Patch(color=colors[i], label=labels[values[i]]) \
                for i in range(len(values)) if counts[i] > 0.05]
        plt.legend(handles=patches)


def show_examples(generator):
    x, y = generator.__next__()
    for i in range(0,3):
        image = x[i]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.argmax(y[i], axis=2)
        plt.subplot(1,2,1)
        plt.imshow(image.astype('uint8'))
        plt.subplot(1,2,2)
        plt.imshow(mask, cmap='viridis')
        plt.show()
    return x, y


def preprocess_data(img, mask, num_class,backbone):
    preprocess_input = sm.get_preprocessing(backbone)
    img = preprocess_input(img)
    mask = to_categorical(mask, num_class)
    return (img, mask)

def trainGenerator(train_img_path, train_mask_path, num_class, backbone, seed, batchsize, size=(64, 64), aug=True, show_raw=False):
    
    img_data_gen_args = dict(horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='reflect')
    
    if aug:
        image_datagen = ImageDataGenerator(**img_data_gen_args)
        mask_datagen = ImageDataGenerator(**img_data_gen_args)
    else:
        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()
    
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode=None,
        batch_size=batchsize,
        seed=seed,
        target_size=size)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode=None,
        color_mode='grayscale',
        batch_size=batchsize,
        seed=seed,
        target_size=size)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        if show_raw:
            raw_img = np.copy(img)
            preprocess_input = sm.get_preprocessing(backbone)
            img = preprocess_input(img)
            mask = to_categorical(mask, num_class)
            img = np.stack((img, raw_img), axis=-1)
            yield (img, mask)
        else:
            preprocess_input = sm.get_preprocessing(backbone)
            img = preprocess_input(img)
            mask = to_categorical(mask, num_class)
            yield (img, mask)