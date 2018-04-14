# -*- coding: utf-8 -*-
from datetime import datetime
from glob import glob
import os, shutil
import numpy as np
import skimage.io
import skimage.transform
from skimage.transform import AffineTransform, warp
from skimage.transform import resize, SimilarityTransform

# 이미지를 읽어 들임 
def load(paths_train):
    images = []
    imagenames = []
    labels = []

    for i, path in enumerate(paths_train):
        image = resize(skimage.io.imread(path), (224,224))
        imagename = os.path.basename(path)
        label = os.path.basename(os.path.dirname(path))
        images.append(image)
        imagenames.append(imagename)
        labels.append(label)

    return images, imagenames, labels

# 변환 실행
def fast_warp(img, tf, output_shape=(50, 50), mode='constant', order=1):
    m = tf.params
    return warp(img, m, output_shape=output_shape, mode=mode, order=order)

def build_centering_transform(image_shape, target_shape=(50, 50)):

    if len(image_shape) == 2:
        rows, cols = image_shape
    else:
        rows, cols, _ = image_shape
    trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return SimilarityTransform(translation=(shift_x, shift_y))

def build_center_uncenter_transforms(image_shape):
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5
    tform_uncenter = SimilarityTransform(translation=-center_shift)
    tform_center = SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter

def build_transform(zoom=(1.0, 1.0), rot=0, shear=0, trans=(0, 0), flip=False):
    if flip:
        shear += 180
        rot += 180

    r_rad = np.deg2rad(rot)
    s_rad = np.deg2rad(shear)
    tform_augment = AffineTransform(scale=(1/zoom[0], 1/zoom[1]),
                  rotation=r_rad, shear=s_rad, translation=trans)
    return tform_augment

def random_transform(zoom_range, rotation_range, shear_range,
         translation_range, do_flip=True, allow_stretch=False, rng=np.random):
    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translation = (shift_x, shift_y)

    rotation = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    if do_flip:
        flip = (rng.randint(2) > 0)
    else:
        flip = False

    log_zoom_range = [np.log(z) for z in zoom_range]

    if isinstance(allow_stretch, float):
        log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
        zoom = np.exp(rng.uniform(*log_zoom_range))
        stretch = np.exp(rng.uniform(*log_stretch_range))
        z_x = zoom * stretch
        z_y = zoom / stretch
    elif allow_stretch is True:
        z_x = np.exp(rng.uniform(*log_zoom_range))
        z_y = np.exp(rng.uniform(*log_zoom_range))
    else:
        z_x = z_y = np.exp(rng.uniform(*log_zoom_range))

    return build_transform((z_x, z_y), rotation, shear, translation, flip)

# 변환 시 사용하는 파라미터를 준비
def perturb(img, augmentation_params, t_shape=(50, 50), rng=np.random):
    tf_centering = build_centering_transform(img.shape, t_shape)
    tf_center, tf_uncenter = build_center_uncenter_transforms(img.shape)
    tf_aug = random_transform(rng=rng, **augmentation_params)
    tf_aug = tf_uncenter + tf_aug + tf_center
    tf_aug = tf_centering + tf_aug
    warp_one = fast_warp(img, tf_aug, output_shape=t_shape, mode='constant')
    return warp_one.astype('float32')




# 메인 디렉터리 작성
path_root = '../../data/Caltech-101'

shutil.rmtree('../../data/Caltech-101/train_org')

# train 디렉터리를 다른 이름으로 바꿈
os.rename("../../data/Caltech-101/train", "../../data/Caltech-101/train_org")

# valid 디렉터리를 다른 이름으로 바꿈
os.rename("../../data/Caltech-101/valid", "../../data/Caltech-101/valid_org")

# test 디렉터리를 다른 이름으로 바꿈.
os.rename("../../data/Caltech-101/test", "../../data/Caltech-101/test_org")

# 디렉터리가 존재하지 않는 경우 새로 작성
if not os.path.exists('../../data/Caltech-101/train/all'):
    os.makedirs('../../data/Caltech-101/train/all')
if not os.path.exists('../../data/Caltech-101/valid/all'):
    os.makedirs('../../data/Caltech-101/valid/all')
if not os.path.exists('../../data/Caltech-101/test/all'):
    os.makedirs('../../data/Caltech-101/test/all')
for ho in range(0, 2):
    for aug in xrange(5):
        if not os.path.exists('../../data/Caltech-101/train/%i/%i'%(ho, aug)):
            os.makedirs('../../data/Caltech-101/train/%i/%i'%(ho, aug))
        if not os.path.exists('../../data/Caltech-101/valid/%i/%i'%(ho,aug)):
            os.makedirs('../../data/Caltech-101/valid/%i/%i'%(ho,aug))
        if not os.path.exists('../../data/Caltech-101/test/%i'%aug):
            os.makedirs('../../data/Caltech-101/test/%i'%aug)

# data_augmentation 파라미터
augmentation_params = {
    # 확장/축소(가로세로 비율로 고정)
    'zoom_range': (1 / 1, 1),
    # 회전 각도
    'rotation_range': (-15, 15),
    # 밀어 당기기
    'shear_range': (-20, 20),
    # 평행 이동
    'translation_range': (-30, 30),
    # 반전
    'do_flip': False,
    # 늘리기/줄이기(가로세로 비율을 고정하지 않음)
    'allow_stretch': 1.3,
}

# HoldOut을 2회 반복 
for ho in xrange(0, 2):

    paths_train = sorted(glob('%s/train_org/%i/*/*.jpg'%(path_root, ho)))
    paths_valid = sorted(glob('%s/valid_org/%i/*/*.jpg'%(path_root, ho)))
    paths_test = sorted(glob('%s/test_org/*/*.jpg'%path_root))

    # 이미지를 읽어 들임
    images_train, imagenames_train, labels_train = load(paths_train)
    images_valid, imagenames_valid, labels_valid = load(paths_valid)
    images_test, imagenames_test, labels_test = load(paths_test)


    # 5배로 증가하므로 5번 반복. 
    for s in xrange(5):
        seed = ho * 5 + s
        np.random.seed(seed)

        # train 데이터를 준비
        path_output = '%s/train/%i/%i'%(path_root, ho, s)

        # 디렉터리를 작성
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        # 이미지 각각에 대해 반복 
        for i, image in enumerate(images_train):
            path_dir = os.path.join(path_output, labels_train[i])
            all_path_dir = os.path.join(path_root, 'train/all', labels_train[i])
            if not os.path.exists(path_dir):
                os.mkdir(path_dir)
            if not os.path.exists(all_path_dir):
                os.makedirs(all_path_dir)
            name = imagenames_train[i]
            # augmentation 실행
            image = perturb(image, augmentation_params, (224, 224))
            skimage.io.imsave(os.path.join(path_dir, name), image)
            # augment 데이터를 하나의 디렉터리에 정리
            # ResNet용
            path_output_tmp = '%s/train/all/'%(path_root)
            path_dir_tmp = os.path.join(path_output_tmp, labels_train[i])
            name_tmp = name.split(".")[0] + "_" + str(seed) + "." + name.split(".")[1]
            skimage.io.imsave(os.path.join(path_dir_tmp, name_tmp), image)



        # valid 데이터를 준비
        path_output = '%s/valid/%i/%i'%(path_root, ho, s)

        # 디렉터리를 작성
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        # 이미지 각각에 대해서 반복 
        for i, image in enumerate(images_valid):
            path_dir = os.path.join(path_output, labels_valid[i])
            all_path_dir = os.path.join(path_root, 'valid/all', labels_valid[i])
            if not os.path.exists(path_dir):
                os.mkdir(path_dir)
            if not os.path.exists(all_path_dir):
                os.makedirs(all_path_dir)
            name = imagenames_valid[i]
            # augmentation 실행
            image = perturb(image, augmentation_params, (224, 224))
            skimage.io.imsave(os.path.join(path_dir, name), image)
            # augmentしたデータを一つのディレクトリにまとめる。
            # ResNet용
            path_output_tmp = '%s/valid/all/'%(path_root)
            path_dir_tmp = os.path.join(path_output_tmp, labels_valid[i])
            name_tmp = name.split(".")[0] + "_" + str(seed) + "." + name.split(".")[1]
            skimage.io.imsave(os.path.join(path_dir_tmp, name_tmp), image)

        if ho == 0:

            # test 데이터를 준비
            path_output = '%s/test/%i'%(path_root, s)

            # 디렉토리를 작성
            if not os.path.exists(path_output):
                os.makedirs(path_output)

            # 이미지 각각에 대해서 반복
            for i, image in enumerate(images_test):
                path_dir = os.path.join(path_output, labels_test[i])
                all_path_dir = os.path.join(path_root, 'test/all', labels_test[i])
                if not os.path.exists(path_dir):
                    os.mkdir(path_dir)
                if not os.path.exists(all_path_dir):
                    os.makedirs(all_path_dir)
                name = imagenames_test[i]
                # augmentation실행
                image = perturb(image, augmentation_params, (224, 224))
                skimage.io.imsave(os.path.join(path_dir, name), image)
                # augment 데이터를 하나의 디렉터리에 정리
                # ResNet용
                path_output_tmp = '%s/test/all/'%(path_root)
                path_dir_tmp = os.path.join(path_output_tmp, labels_test[i])
                name_tmp = name.split(".")[0] + "_" + str(seed) + "." + name.split(".")[1]
                skimage.io.imsave(os.path.join(path_dir_tmp, name_tmp), image)

