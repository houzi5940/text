#!/usr/bin/env python
# coding: utf-8


from tensorflow.keras import datasets, layers, models
import tensorflow as tf


import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers


def makeDatasets(path):

    allImages = []
    allLabels = []
#     subdirs=os.listdir('./dataset')
    subdirs = os.listdir(path)
    subdirs.sort()
    print(subdirs)
    classes = len(subdirs)
    for subdir in range(classes):
        for index in os.listdir(os.path.join(path, subdirs[subdir])):
            # rint(index)
            imagePath = os.path.join(path, subdirs[subdir], index)
            #  print(indexdir)
            # print(imagePath)
            try:

                img = cv2.imread(imagePath)
                img = cv2.resize(img, dsize=(32, 32),
                                 interpolation=cv2.INTER_AREA)
                allImages.append(img)
                allLabels.append(subdir)
            except:
                print(imagePath)
                continue
                # print("出错了："+imagePath)

    print("完成")
    c = list(zip(allImages, allLabels))
    random.shuffle(c)
    allImages, allLabels = zip(*c)

    trainNum = int(0.9*len(allImages))
    print(trainNum)
    trainImages, trainLabels = allImages[:trainNum], allLabels[:trainNum]
    testImages, testLabels = allImages[trainNum:], allLabels[trainNum:]

    print(np.array(trainImages).shape)
    print(np.array(testImages).shape)
    print(np.array(trainLabels).shape)
    print(np.array(testLabels).shape)

    np.save("project01-ljfl_train_images.npy", trainImages)
    np.save("project01-ljfl_ljfl_test_images.npy", testImages)
    np.save("project01-ljfl_train_labels.npy", trainLabels)
    np.save("project01-ljfl_test_labels.npy", testLabels)

    return trainImages, trainLabels, testImages, testLabels


def loadDatesets(path):
    try:
        trainImages = np.load("./all_data/project01-ljfl_train_images.npy")
        testImages = np.load("./all_data/project01-ljfl_ljfl_test_images.npy")
        trainLabels = np.load("./all_data/project01-ljfl_train_labels.npy")
        testLabels = np.load("./all_data/project01-ljfl_test_labels.npy")
    except:
        trainImages, trainLabels, testImages, testLabels = makeDatasets(path)

    return trainImages, trainLabels, testImages, testLabels

    # 主函数


# #绘制数据集中的图片
# def plot_image(image):
#     b,g,r = cv2.split(image)
#     img_rgb = cv2.merge([r,g,b])

#     fig=plt.gcf()
#     fig.set_size_inches(2,2)
#     plt.imshow(img_rgb)
#     plt.show()


# 创建模型


def cnnModel():
    model = models.Sequential()
    model.add(layers.Conv2D(
        48, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(10, activation='softmax'))

    return model


# # # In[51]:


# 绘制训练过程曲线变化图
def drawtrainning(history):
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


# if __name__ == "__main__":
#     path=r'./picture'
#     logdir = os.path.join("./save_weights")
#     output_model_file = os.path.join(logdir, "model50.h5")
#     print(output_model_file)
#
#     trainImages ,trainLabels, testImages, testLabels=loadDatesets(path)
#
#     trainImages=trainImages/255.0
#     testImages=testImages/255.0
#
#
#     model=cnnModel()
#
#
#
#

#     opt=optimizers.Adam(lr=1e-8)
#     model.compile(optimizer=opt,
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])


#     cp_callback = tf.keras.callbacks.ModelCheckpoint(
# #                         filepath=output_model_file,
# #                         monitor='val_loss',
# #                         verbose=1,
# #                         save_best_only=True,
# #                         save_weights_only=True,
# #                         mode='min'
# #                         )
#
#
# #     history = model.fit(trainImages, trainLabels, epochs=10, batch_size=8,
# #                     validation_split=0.1,callbacks=[cp_callback])
#
# # 模型
# #         model.compile(optimizer='adam',
# #                   loss='sparse_categorical_crossentropy',
# #                   metrics=['acc'])
# #
# #         history = model.fit(x=trainImages, y=trainLabels, epochs=10, batch_size=8,
# #                     validation_split=0.1)
# #
# #
# #     # 模型存放路径
# #         save_path = './save_weights/model50.h5'
# #         model.save_weights(save_path)
# #
# #
# #
# #
# #         drawtrainning(history)
# #
# #
# #         results=model.evaluate(testImages,testLabels)
# #         print("test loss:", results[0],"test acc:", results[1])
#
#
#
#
# y=model.predict_classes(testImages)
#
# print(y[100:120])
# print(testLabels[100:120])
# #加载模型权重
#     try:
#         model.load_weights(output_model_file)
#         print("Load the existing model parameters successfully, continue training")
#     except:
#         print("No model parameter file, start to train")

def shuiguofenlei(img):
    weight_file = "D:\\FLASK\\api_service\\detect_apis\\model50.h5"
    model = cnnModel()
    if os.path.exists(weight_file):
        model.load_weights(weight_file)
    else:
        print("trained weight file is not exist")
    data = cv2.resize(img, dsize=(32, 32))
    data = np.reshape(data, (1, 32, 32, 3))
    prediction = model.predict_classes(data)[0]
    return prediction
