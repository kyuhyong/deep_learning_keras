# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import permutation

import os, glob, cv2, math, sys
import pandas as pd

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
import keras
kerasVersion = keras.__version__[0]
print('Using keras version :'+kerasVersion)
if kerasVersion > 1:
    from keras.layers.convolutional import Conv2D, MaxPooling2D
else :
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
# seed 값
np.random.seed(1)

# 사용하는 이미지 사이즈 
img_rows, img_cols = 224, 224

# 이미지 데이터 1장을 읽어들이고 리사이즈를 함.
def get_im(path):

    img = cv2.imread(path)
    resized = cv2.resize(img, (img_cols, img_rows))

    return resized


# 데이터를 읽어들이고 정규화 및 셔플을 수행함.
def read_train_data(ho=0, kind='train'):

    train_data = []
    train_target = []

    # 학습용 데이터 읽어들이기 
    for j in range(0, 6): # 0～5까지

        path = '../../data/Caltech-101/'
        path += '%s/%i/%i/*.jpg'%(kind, ho, j)

        files = sorted(glob.glob(path))
        if not os.path.exists(path):
            print(path+' 폴더가 존재하지 않습니다!')
            exit()
        for fl in files:

            flbase = os.path.basename(fl)

            # 이미지 1장 읽어들이기 
            img = get_im(fl)
            img = np.array(img, dtype=np.float32)

            # 정규화(GCN) 실행
            img -= np.mean(img)
            img /= np.std(img)

            train_data.append(img)
            train_target.append(j)

    # 읽어들인 데이터를 numpy의 array로 변환
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.uint8)
    # (레코드수, 종, 횡, 채널수)를 (레코드수, 채널수, 종, 횡)으로 변환. 케라스 버전 1만 해당
    if kerasVersion == 1:
        train_data = train_data.transpose((0, 3, 1, 2))

    # 타깃을 6차원의 데이터로 변환 。
    # 예) 1 -> 0,1,0,0,0,0   2 -> 0,0,1,0,0,0
    train_target = np_utils.to_categorical(train_target, 6)

    # 데이터를 셔플
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]

    return train_data, train_target


# 테스트 데이터를 읽어들임.
def load_test(test_class, aug_i):

    #path = '../../data/Caltech-101/test/%i/%i/*.jpg'%(aug_i, test_class)
    path = '../../data/Caltech-101/test/%i/*.jpg'%(test_class)

    files = sorted(glob.glob(path))
    if not os.path.exists(path):
        print(path+' 폴더가 존재하지 않습니다!')
    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        #print(os.path.basename(fl))
        img = get_im(fl)
        img = np.array(img, dtype=np.float32)

        # 정규화(GCN) 실행
        img -= np.mean(img)
        img /= np.std(img)

        X_test.append(img)
        X_test_id.append(flbase)

    # 읽어들인 데이터를 numpy의 array로 변환 
    test_data = np.array(X_test, dtype=np.float32)
    # (레코드수, 종, 횡, 채널수)를 (레코드수, 채널수, 종, 횡)으로 변환. 케라스 버전 1만 해당
    if kerasVersion == 1:
        test_data = test_data.transpose((0, 3, 1, 2))
    #test_data = test_data.transpose((0, 1, 2, 3))

    return test_data, X_test_id


# 9층 CNN 모델 구축
def layer_9_model():
    # Keras의 Sequential을 기초 모델로 사용 ---①
    model = Sequential()
    if kerasVersion > 1:
        # 합성층(Convolution)을 모델에 추가 ---②
        model.add(Conv2D(32, (3, 3), padding='same', activation='linear',
         input_shape=(img_rows, img_cols, 3)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Conv2D(32, (3, 3), padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        # 풀링층(MaxPooling)을 모델에 추가 ---③
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Conv2D(64, (3, 3), padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Conv2D(128, (3, 3), padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    else :
        # 합성층(Convolution)을 모델에 추가 ---②
        model.add(Convolution2D(32, 3, 3, border_mode='same', activation='linear',
         input_shape=(3, img_rows, img_cols)))
        model.add(LeakyReLU(alpha=0.3))

        model.add(Convolution2D(32, 3, 3, border_mode='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        # 풀링층(MaxPooling)을 모델에 추가 ---③
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

        model.add(Convolution2D(128, 3, 3, border_mode='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Convolution2D(128, 3, 3, border_mode='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    # Flatten층을 모델에 추가 -- ④
    model.add(Flatten())
    # 전결합층(Dense)을 모델에 추가 --- ⑤
    model.add(Dense(1024, activation='linear'))
    model.add(LeakyReLU(alpha=0.3))
    # Dropout층을 모델에 추가 --- ⑥
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='linear'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    # 최종 아웃풋을 구축 --- ⑦
    model.add(Dense(6, activation='softmax'))

    # 손실 함수와 기울기의 계산에 사용하는 식을 정의한다. -- ⑧
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
             loss='categorical_crossentropy', metrics=["accuracy"])
    return model


# 모델의 구조과 가중치를 읽어들임
def read_model(ho, modelStr='', epoch='00'):
    # 모델 구조의 파일명 
    json_name = 'architecture_%s_%i.json'%(modelStr, ho)
    # 모델 가중치의 파일명
    weight_name = 'model_weights_%s_%i_%s.h5'%(modelStr, ho, epoch)

    # 모델 구조를 json으로부터 읽어들여서 모델 오브젝트로 변환
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    # 모델 오브젝트에 가중치를 읽어들임.
    model.load_weights(os.path.join('cache', weight_name))

    return model


# 모델의 구조를 저장
def save_model(model, ho, modelStr=''):
    # 모델 오브젝트를 json 형식으로 변환
    json_string = model.to_json()
    # current 디렉터리로 cashe 디렉터리가 없으면 새로 만듦.
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    # 모델 구조를 저장하기 위한 파일명 
    json_name = 'architecture_%s_%i.json'%(modelStr, ho)
    # 모델 구조를 저장
    open(os.path.join('cache', json_name), 'w').write(json_string)


def run_train(modelStr=''):
    # HoldOut를 2번 실행.
    for ho in range(2):
        # 모델 구축
        model = layer_9_model()
        # train 데이터 읽어들임
        t_data, t_target = read_train_data(ho, 'train')
        v_data, v_target = read_train_data(ho, 'valid')
        # CheckPoint를 설정。에폭마다 가중치를 저장 
        cp = ModelCheckpoint('./cache/model_weights_%s_%i_{epoch:02d}.h5'%(modelStr, ho),
        monitor='val_loss', save_best_only=False)
        # train 실행
        model.fit(t_data, t_target, batch_size=64,
                  nb_epoch=40,
                  verbose=1,
                  validation_data=(v_data, v_target),
                  shuffle=True,
                  callbacks=[cp])
        # 모델 구조의 저장
        save_model(model, ho, modelStr)

# 테스트 데이터의 클래스를 예측
def run_test(modelStr, epoch1, epoch2):

    # 클래스명 얻기
    columns = []
    for line in open("../../data/Caltech-101/label.csv", 'r'):
        sp = line.split(',')
        for column in sp:
            columns.append(column.split(":")[1])

    # 테스트 데이터가 각 클래스로 나누어지므로
    # 1클래스씩 읽어들여서 예측을 실행. 
    idx=0    
    for test_class in range(0, 6):
        yfull_test = []
        print(columns[idx])
        idx+=1
        # 데이터 확장이 처리된 이미지를 읽어들이기 위해서 5번 반복 
        for aug_i in range(0,5):
            # 테스트 데이터를 읽어들임. 
            test_data, test_id = load_test(test_class, aug_i)
            # HoldOut 2번 반복
            for ho in range(2):

                if ho == 0:
                    epoch_n = epoch1
                else:
                    epoch_n = epoch2
                # 학습이 끝난 모델을 읽어들임
                model = read_model(ho, modelStr, epoch_n)
                # 예측을 실행
                test_p = model.predict(test_data, batch_size=128, verbose=1)
                yfull_test.append(test_p)
        # 예측 결과를 평균
        test_res = np.array(yfull_test[0])
        for i in range(1,10):
            test_res += np.array(yfull_test[i])
        test_res /= 10
        # 예측 결과와 클래스명, 이미지명을 합함.. 
        result1 = pd.DataFrame(test_res, columns=columns)
        result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
        # 순서를 바꿈
        result1 = result1.ix[:,[6, 0, 1, 2, 3, 4, 5]]

        if not os.path.isdir('subm'):
            os.mkdir('subm')
        sub_file = './subm/result_%s_%i.csv'%(modelStr, test_class)
        # 최종 예측 결과를 출력. 
        result1.to_csv(sub_file, index=False)
        # 예측의 정밀도를 추정 
        # 가장 큰 값이 들어 있는 열이 test_class인 레코드를 찾음. 
        one_column = np.where(np.argmax(test_res, axis=1)==test_class)
        print ("정답수　　" + str(len(one_column[0])))
        print ("오답수　" + str(test_res.shape[0] - len(one_column[0])))

# 실행 프로그램을 호출
if __name__ == '__main__':
    # 인수를 얻음
    # param[1] = train or test
    # param[2] = test 실행 시에만 사용 에폭수 1
    # param[3] = test 실행 시에만 사용 에폭수 2
    param = sys.argv
    if len(param) < 2:
        sys.exit ("Usage: python 9_Layer_CNN.py [train, test] [1] [2]")
    # train or test
    run_type = param[1]
    if run_type == 'train':
        run_train('9_Layer_CNN')
    elif run_type == 'test':
        # test의 경우、사용하는 에폭수를 인수로부터 얻음. 
        if len(param) == 4:
            epoch1 = "%02d"%(int(param[2])-1)
            epoch2 = "%02d"%(int(param[3])-1)
            run_test('9_Layer_CNN', epoch1, epoch2)
        else:
            sys.exit ("Usage: python 9_Layer_CNN.py [train, test] [1] [2]")
    else:
        sys.exit ("Usage: python 9_Layer_CNN.py [train, test] [1] [2]")


