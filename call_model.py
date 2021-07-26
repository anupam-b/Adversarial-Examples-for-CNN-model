from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, experimental,Input,Dropout,Concatenate
import numpy as np
import pandas as pd
import os
import cv2
from scipy import ndimage as ndi
from skimage.feature import canny

def build_cnn(input_shape):

    inpt = Input(shape=input_shape)
    
    conv_1 = Convolution2D(32, (5, 5), padding='same', activation='relu')(inpt)
    drop_1 = Dropout(rate=0.2)(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(drop_1)

    conv_2 = Convolution2D(64, (5, 5), padding='same', activation='relu')(pool_1)
    drop_2 = Dropout(rate=0.3)(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(drop_2)

    conv_3 = Convolution2D(128, (5, 5), padding='same', activation='relu')(pool_2)
    drop_3 = Dropout(rate=0.4)(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(drop_3)

    concat = Concatenate(axis=-1)([Flatten()(pool_1), Flatten()(pool_2), Flatten()(pool_3)])
    dense_1 = Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))(concat)
    drop_4 = Dropout(rate=0.5)(dense_1)
    output = Dense(43, activation=None, kernel_regularizer=keras.regularizers.l2(0.0001))(drop_4)

    model = keras.models.Model(inputs=inpt, outputs=output)

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def load_model_weights(problem):
    filename = os.path.join(problem)
    model=build_cnn((32, 32, 3))
    model.load_weights(filename)
    print("\nModel weights successfully loaded\n")
    return model

def match_pred_yd(predictions):
    dataframe = pd.read_csv("sign_name.csv")
    l = []
    for i in range(len(predictions)):
        l.append(int(dataframe['ClassId'].loc[dataframe['ModelId'] == predictions[i]]))
    return np.array(l)

def match_pred_ym(predictions):
    dataframe = pd.read_csv("sign_name.csv")
    l = []
    for i in range(len(predictions)):
        l.append(int(dataframe['ModelId'].loc[dataframe['ClassId'] == predictions[i]]))
    return l


def save_in_distribution_attack(model, attack_type, is_target, class_id, x, result):

    csv_data_attack = pd.DataFrame(columns=['path_adversarial', 'original_prevision', 'adversarial_prevision', 'success'])
    size = len(x)

    for i in range(len(x)):

        original_image = x[i]
        img_orig = np.expand_dims(original_image, axis=0)
        res_orig = model.predict(img_orig)
        Ypred_orig = np.argmax(res_orig, axis=1)

        if(attack_type == "FG"):
            adv1 = result[i]
            img_adv1 = np.expand_dims(adv1, axis=0)
            res_adv1 = model.predict(img_adv1)
            Ypred_adv1 = np.argmax(res_adv1, axis=1)

            if(is_target):
                if(int(Ypred_adv1[0]) != int(class_id)):
                    csv_data_attack.loc[i] = ["Adversarial_img/in_distribution/fg/"+"target/" + str(i) + ".png", Ypred_orig[0], Ypred_adv1[0], 0]
                else:
                    csv_data_attack.loc[i] = ["Adversarial_img/in_distribution/fg/"+"target/" + str(i) + ".png", Ypred_orig[0], Ypred_adv1[0], 1]
                cv2.imwrite("Adversarial_img/in_distribution/fg/"+"target/" + str(i) + '.png', adv1 * 255)
            else:
                if(int(Ypred_adv1[0]) != int(Ypred_orig[0])):
                    csv_data_attack.loc[i] = ["Adversarial_img/in_distribution/fg/"+"untarget/" + str(i) + ".png", Ypred_orig[0], Ypred_adv1[0], 1]
                else:
                    csv_data_attack.loc[i] = ["Adversarial_img/in_distribution/fg/"+"untarget/" + str(i) + ".png", Ypred_orig[0], Ypred_adv1[0], 0]
                cv2.imwrite("Adversarial_img/in_distribution/fg/"+"untarget/" + str(i) + '.png', adv1 * 255)

        else:
            adv2 = result[i]
            img_adv2 = np.expand_dims(adv2, axis=0)
            res_adv2 = model.predict(img_adv2)
            Ypred_adv2 = np.argmax(res_adv2, axis=1)

            if(is_target):
                if(int(Ypred_adv2[0]) != int(class_id)):
                    csv_data_attack.loc[i] = ["Adversarial_img/in_distribution/it/"+"target/" + str(i) + ".png", Ypred_orig[0], Ypred_adv2[0], 0]
                else:
                    csv_data_attack.loc[i] = ["Adversarial_img/in_distribution/it/"+"target/" + str(i) + ".png", Ypred_orig[0], Ypred_adv2[0], 1]

                cv2.imwrite("Adversarial_img/in_distribution/it/"+"target/" + str(i) + '.png', adv2 * 255)

            else:
                if(int(Ypred_adv2[0]) != int(Ypred_orig[0])):
                    csv_data_attack.loc[i] = ["Adversarial_img/in_distribution/it/"+"untarget/" + str(i) + ".png", Ypred_orig[0], Ypred_adv2[0], 1]
                else:
                    csv_data_attack.loc[i] = ["Adversarial_img/in_distribution/it/"+"untarget/" + str(i) + ".png", Ypred_orig[0], Ypred_adv2[0], 0]
                cv2.imwrite("Adversarial_img/in_distribution/it/"+"untarget/" + str(i) + '.png', adv2 * 255)

    count = 0
    for index, row in csv_data_attack.iterrows():
        if (row['success'] != 1):
            count += 1

    accuracy = round((1 - (float(count) / float(size))) * 100, 2)

    print("Attacks successful: " + str(accuracy) + "% ")

    if(attack_type == "FG"):
        if(is_target):
            csv_data_attack.to_csv("Adversarial_img/in_distribution/fg/"+"target/result.csv", sep=',', index=False)
        else:
            csv_data_attack.to_csv("Adversarial_img/in_distribution/fg/"+"untarget/result.csv", sep=',', index=False)
    else:
        if(is_target):
            csv_data_attack.to_csv("Adversarial_img/in_distribution/it/"+"target/result.csv", sep=',', index=False)
        else:
            csv_data_attack.to_csv("Adversarial_img/in_distribution/it/"+"untarget/result.csv", sep=',', index=False)


def save_out_distribution_attack(model, attack_type, class_id, method, x, result):
    
    csv_data_attack = pd.DataFrame(columns=['path_adversarial', 'adversarial_prevision', 'success'])
    size = len(x)

    for i in range(len(x)):

        if(attack_type == "FG"):
            adv1 = result[i]
            img_adv1 = np.expand_dims(adv1, axis=0)
            res_adv1 = model.predict(img_adv1)
            Ypred_adv1 = np.argmax(res_adv1, axis=1)

            if(method=="LOGO"):
                if(int(Ypred_adv1[0]) != int(class_id)):
                    csv_data_attack.loc[i] = ["Adversarial_img/out_distribution/logo_signs/fg/target/" + str(i) + ".png", Ypred_adv1[0], 0]
                else:
                    csv_data_attack.loc[i] = ["Adversarial_img/out_distribution/logo_signs/fg/target/" + str(i) + ".png", Ypred_adv1[0], 1]
                cv2.imwrite("Adversarial_img/out_distribution/logo_signs/fg/target/" + str(i) + '.png', adv1 * 255)
            else:
                if(int(Ypred_adv1[0]) != int(class_id)):
                    csv_data_attack.loc[i] = ["Adversarial_img/out_distribution/blank_signs/fg/target/" + str(i) + ".png", Ypred_adv1[0], 0]
                else:
                    csv_data_attack.loc[i] = ["Adversarial_img/out_distribution/blank_signs/fg/target/" + str(i) + ".png", Ypred_adv1[0], 1]
                cv2.imwrite("Adversarial_img/out_distribution/blank_signs/fg/target/" + str(i) + '.png', adv1 * 255)
        else:
            adv2 = result[i]
            img_adv2 = np.expand_dims(adv2, axis=0)
            res_adv2 = model.predict(img_adv2)
            Ypred_adv2 = np.argmax(res_adv2, axis=1)

            if(method=="LOGO"):
                if(int(Ypred_adv2[0]) != int(class_id)):
                    csv_data_attack.loc[i] = ["Adversarial_img/out_distribution/logo_signs/iterative/target/" + str(i) + ".png", Ypred_adv2[0], 0]
                else:
                    csv_data_attack.loc[i] = ["Adversarial_img/out_distribution/logo_signs/iterative/target/" + str(i) + ".png", Ypred_adv2[0], 1]
                cv2.imwrite("Adversarial_img/out_distribution/logo_signs/iterative/target/" + str(i) + '.png', adv2 * 255)
            else:
                if(int(Ypred_adv2[0]) != int(class_id)):
                    csv_data_attack.loc[i] = ["Adversarial_img/out_distribution/blank_signs/iterative/target/" + str(i) + ".png", Ypred_adv2[0], 0]
                else:
                    csv_data_attack.loc[i] = ["Adversarial_img/out_distribution/blank_signs/iterative/target/" + str(i) + ".png", Ypred_adv2[0], 1]
                cv2.imwrite("Adversarial_img/out_distribution/blank_signs/iterative/target/" + str(i) + '.png', adv2 * 255)

    count = 0
    for index, row in csv_data_attack.iterrows():
        if (row['success'] != 1):
            count += 1

    accuracy = round((1 - (float(count) / float(size))) * 100, 2)

    print("Attacks successful: " + str(accuracy) + "% ")

    if(attack_type == "FG"):
        if(method=="LOGO"):
            csv_data_attack.to_csv("Adversarial_img/out_distribution/logo_signs/fg/target/"+"result.csv", sep=',', index=False)
        else:
            csv_data_attack.to_csv("Adversarial_img/out_distribution/blank_signs/fg/target/"+"result.csv", sep=',', index=False)
    else:
        if(method=="LOGO"):
            csv_data_attack.to_csv("Adversarial_img/out_distribution/logo_signs/iterative/target/"+"result.csv", sep=',', index=False)
        else:
            csv_data_attack.to_csv("Adversarial_img/out_distribution/blank_signs/iterative/target/"+"result.csv", sep=',', index=False)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    if iteration == total: 
        print()

def resize(image, size=(32,32), interp='bilinear'):
    img = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    img = (img / 255.).astype(np.float32)
    return img

def resize_all(images, interp='bilinear'):
    if images[0].ndim == 3:
        shape = (len(images),) + (32,32) + (3,)
    elif images[0].ndim == 2:
        shape = (len(images),) + (32,32)
    else:
        return
    images_rs = np.zeros(shape)
    for i, image in enumerate(images):
        images_rs[i] = resize(image, interp=interp)
    return images_rs

def find_sign_area(image, sigma=1):
    edges = canny(image, sigma=sigma)
    fill = ndi.binary_fill_holes(edges)
    label_objects, _ = ndi.label(fill)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = np.zeros_like(sizes)
    sizes[0] = 0
    mask_sizes[np.argmax(sizes)] = 1.
    cleaned = mask_sizes[label_objects]

    return cleaned

def read_images(path, resize=False, interp='bilinear'):
    imgs = []
    valid_images = [".jpg", ".gif", ".png", ".tga", ".jpeg", ".ppm"]
    for f in sorted(os.listdir(path)):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        im = cv2.imread(os.path.join(path, f))
        if resize:
            im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_LINEAR)
        im = (im / 255.).astype(np.float32)
        imgs.append(im)
    return np.array(imgs)

def read_labels(path):
    with open(path) as f:
        content = f.readlines()
    content = [int(x.strip()) for x in content]
    return content

def load_out_samples(img_dir):

    images = read_images(img_dir, True)
    masks_full = []

    for i, image in enumerate(images):
        mask = find_sign_area(rgb2gray(image))
        masks_full.append(mask)

    masks = resize_all(masks_full, interp='nearest')
    x_ben = resize_all(images, interp='bilinear')

    return x_ben, masks

def load_samples(img_dir, label_path, tg):

    images = read_images(img_dir, True)
    masks_full = []

    labels = read_labels(label_path)
    result = match_pred_ym(labels)

    rm_indx = -1
    if(tg in result):
        print("Deleting target class from samples...")
        rm_indx = result.index(tg)
        images = np.delete(images, [rm_indx], axis=0)
        result = np.delete(result, [rm_indx], axis=0)


    for i, image in enumerate(images):
        mask = find_sign_area(rgb2gray(image))
        masks_full.append(mask)

    masks = resize_all(masks_full, interp='nearest')
    x_ben = resize_all(images, interp='bilinear')

    return x_ben, result, masks

def rename_signs(res):
    dataframe = pd.read_csv("sign_name.csv")
    r = dataframe['SignName'].loc[dataframe['ModelId'] == res]
    r = r.to_string(index=False, header=False)
    return r.strip()

def resize_image(image, size=32):
    img = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    img = (img / 255.).astype(np.float32)
    return img

def rgb2gray(image):
    if image.ndim == 3:
        return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2])
    elif image.ndim == 4:
        return (0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] +
                0.114 * image[:, :, :, 2])