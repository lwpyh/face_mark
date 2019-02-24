# coding: utf8
import pickle
import random
import numpy as np
from keras import Sequential, Model
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Dense

def load_data():
    with open("training.pkl", "rb") as fi:
        return pickle.load(fi)

def split_data(faces_with_label, train_percent = 0.8):
    pos = int(len(faces_with_label) * train_percent)
    print("Split pos: ", pos)
    x = [item['enc'] for item in faces_with_label]
    print("enc size: ", x[0].shape)
    y = [item['score'] for item in faces_with_label]
    x = np.array(x)
    y = np.array(y)
    # img_rows,img_cols=28,28
    print("shape of x: ", x.shape)
    print("shape of y: ", y.shape)
    train_x, train_y = x[:pos], y[:pos]
    # train_x=train_x.reshape(pos,1,img_rows,img_cols)
    val_x, val_y = x[:pos], y[:pos]
    # print(val_x)
    # val_x =val_x.reshape((len(faces_with_label)-pos), -1, img_rows, img_cols)
    return (train_x, train_y), (val_x, val_y)

def build_model(input_dim,hidden_size):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=input_dim, kernel_initializer="normal", activation="relu"))
    model.add(Dense(hidden_size, input_dim=hidden_size, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

def build_alexnet():
    # 创建模型序列
    model = Sequential()
    # 第一层卷积网络，使用96个卷积核，大小为11x11步长为4， 要求输入的图片为227x227， 3个通道，不加边，激活函数使用relu
    model.add(Conv2D(96, (11, 11), strides=(1, 1), input_shape=(28, 28, 1), padding='same', activation='relu',
                     kernel_initializer='uniform'))  # 池化层 model.
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2))) # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')) #使用池化层，步长为2
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2))) # 第三层卷积，大小为3x3的卷积核使用384个
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')) # 第四层卷积,同第三层
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')) # 第五层卷积使用的卷积核为256个，其他同上
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    return model

def main():
    train_batch_size = 64
    epochs = 1000
    faces_with_score = load_data()
    #random.shuffle(faces_with_score)
    print("Total faces: %d" % len(faces_with_score))
    (train_x, train_y), (val_x, val_y) = split_data(faces_with_score)
    # print(train_y,val_y)
    # model=build_alexnet()
    model=build_model(input_dim=128,hidden_size=512)
    history = model.fit(train_x, train_y, batch_size=train_batch_size, epochs=epochs, shuffle=False, validation_data=(val_x, val_y))
    model.evaluate(val_x, val_y)
    model.save("face_rank_model.h5")


if __name__ == '__main__':
    main()


