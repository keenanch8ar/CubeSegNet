import argparse
import numpy as np
import os
import segmentation_models as sm
import tensorflow as tf
import keras
import datetime

from utils.my_utils import trainGenerator

labels = ['unclassified', 'vegetation', 'no vegetation', 'water', 'cloud']

# Define the available models and their groups

models = {
    'VGG': ['vgg16', 'vgg19'],
    'ResNet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
    'SE-ResNet': ['seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152'],
    'ResNeXt': ['resnext50', 'resnext101'],
    'SE-ResNeXt': ['seresnext50', 'seresnext101'],
    'SENet154': ['senet154'],
    'DenseNet': ['densenet121', 'densenet169', 'densenet201'],
    'Inception': ['inceptionv3', 'inceptionresnetv2'],
    'MobileNet': ['mobilenet', 'mobilenetv2'],
    'EfficientNet': ['efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']
}

# Create a formatted table for the help section
table = '\n'.join(f"{group}:\t{', '.join(models[group])}" for group in models)

# Get the current working directory
current_dir = os.getcwd()

# Construct the paths relative to the current working directory
train_img_path = os.path.join(current_dir, "CubeSegNet_Dataset", "train", "images")
train_mask_path = os.path.join(current_dir, "CubeSegNet_Dataset", "train", "masks")
test_img_path = os.path.join(current_dir, "CubeSegNet_Dataset", "test", "images")
test_mask_path = os.path.join(current_dir, "CubeSegNet_Dataset", "test", "masks")


def main(args):
    BACKBONE = args.backbone
    EPOCHS = args.epochs
    BATCH_SIZE = 12
    SEED = 42
    RAW_N_CLASSES = 16
    N_CLASSES = 5
    
    train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class=N_CLASSES, backbone=BACKBONE,seed=SEED, batchsize=BATCH_SIZE, size=(64, 64))
    test_img_gen = trainGenerator(test_img_path, test_mask_path, num_class=N_CLASSES, backbone=BACKBONE,seed=SEED, batchsize=BATCH_SIZE, size=(64, 64))

    x, y = test_img_gen.__next__() 

    IMG_HEIGHT = x.shape[1]
    IMG_WIDTH  = x.shape[2]
    IMG_CHANNELS = x.shape[3]

    num_train_imgs = len(os.listdir(train_img_path + "/train"))
    num_val_images = len(os.listdir(test_img_path + "/test"))

    steps_per_epoch = num_train_imgs // BATCH_SIZE
    val_steps_per_epoch = num_val_images // BATCH_SIZE

    LR = 0.0001

    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss labels = ['unclassified', 'vegetation', 'no vegetation', 'water', 'cloud']
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.1, 0.2, 0.3, 0.3, 0.1])) 
    focal_loss = sm.losses.BinaryFocalLoss() if N_CLASSES == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

        
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if args.architecture == 'Unet':
            model = sm.Unet(BACKBONE, encoder_weights='imagenet', 
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
                            classes=N_CLASSES, activation='softmax')
        elif args.architecture == 'Linknet':
            model = sm.Linknet(BACKBONE, encoder_weights='imagenet', 
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
                            classes=N_CLASSES, activation='softmax')
        elif args.architecture == 'PSPNet':
            model = sm.PSPNet(BACKBONE, encoder_weights='imagenet', 
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
                            classes=N_CLASSES, activation='softmax')
        elif args.architecture == 'FPN':
            model = sm.FPN(BACKBONE, encoder_weights='imagenet', 
                        input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
                        classes=N_CLASSES, activation='softmax')

        model.compile(optim, total_loss, metrics)


    model.fit(train_img_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=test_img_gen,
                    validation_steps=val_steps_per_epoch,
            )

    # Save the model with a name based on the selected architecture, backbone, and date
    model_name = f"{args.architecture}_{args.backbone}_{datetime.datetime.now().strftime('%d_%m_%y')}"
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(f"{save_dir}/{model_name}/my_seg_model")

    # Convert the saved model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_saved_model(f"{save_dir}/{model_name}/my_seg_model")
    tflite_model = converter.convert()
    with open(f"{save_dir}/{model_name}/{model_name}.tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for semantic segmentation.')
    parser.add_argument('--architecture', type=str, default='Unet', choices=['Unet', 'Linknet', 'PSPNet', 'FPN'], help='Architecture to use for segmentation.')
    parser.add_argument('--backbone', type=str, default='efficientnetb0', choices=sum(models.values(), []), help=f'Backbone model to use for segmentation.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train the model (default: 30)')
    args = parser.parse_args()

    main(args)


