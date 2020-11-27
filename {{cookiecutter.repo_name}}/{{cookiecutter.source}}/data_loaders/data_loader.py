import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from tensorflow.keras.preprocessing.image import random_rotation

class Dataset():
    def __init__(self, cfg):
        if len(cfg)==0: raise TypeError("**config is not given")

        self.filename_list = []
        self.input_size = cfg.input_size # [h, w, c]
        self.nb_train_samples = cfg.nb_train_samples
        self.normalized_range = cfg.normalized_range
        self.shuffle = cfg.shuffle
        self.batch_size = cfg.batch_size
        self.nb_valid_samples = None
        self.use_validation = cfg.use_validation
        valid_rate = cfg.valid_rate

        for imdir in cfg.image_dirs: # tuple内要素(リスト)取り出し
            if type(imdir) is str: # directory path elemが複数ある
                print('\n{} is detected as dataset directory path.'.format(imdir))
                print('\n-----curent dirname -----\n {}'.format(imdir))
                # 再帰的にディレクトリ名のみ取得 os.sep : osに依存しない区切り文字の得方
                inimdir_dirnamelist = [d for d in glob.glob(os.path.join(imdir, '**' + os.sep), recursive=True) if os.path.isdir(d)]
                print('inimdir_namelist in {} : {}'.format(imdir, [dn.replace(imdir, '') for dn in inimdir_dirnamelist]))
                for bottom_imdirname in inimdir_dirnamelist: # 各最下層のファイルリストを取得(画像フォーマットに限る)
                    file_format = ['png', 'jpg', 'bmp']
                    bt_imfilename_list = [flatten for inner in [glob.glob(os.path.join(bottom_imdirname, '*.') + ext) for ext in file_format] for flatten in inner]
                    if len(bt_imfilename_list) != 0: # ディレクトリ内に画像ファイルがある限り
                        self.filename_list.append(bt_imfilename_list) # [[class0], [class1], ...]
                print('\ncomplete.')
            elif type(imdir) is np.ndarray: # WIP
                pass
            elif type(imdir) is 'tfrecord': # WIP
                pass
            else:
                raise ValueError("Dataset class argument in *src has a bad parameter. [str, np.ndarray]")

        self.filename_list = np.array([flatten for inner in self.filename_list for flatten in inner]) # [[], []] -> [] flatten
        im_indices = np.argsort(np.array([os.path.basename(filename) for filename in self.filename_list]))
        self.filename_list = self.filename_list[im_indices]

        self.nb_sample = len(self.filename_list)
        self.nb_valid_sample = int(valid_rate * self.nb_sample) if self.use_validation == True else 0
        self.nb_train_sample = self.nb_sample - self.nb_valid_sample # valid存在しなければnb_train_sample=nb_sample
        print('train sample : {}     valid sample : {}'.format(self.nb_train_sample, self.nb_valid_sample))

        # ((bs, h, w, c), (label,))
        self.data_tensor = tf.data.Dataset.from_tensor_slices((self.filename_list))
        self.data_tensor = self.data_tensor.map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.shuffle: # training dataset -> shuffle, test dataset -> no shuffle
            print('shuffle')
            self.data_tensor = self.data_tensor.shuffle(self.nb_sample, reshuffle_each_iteration=False) # ここでepごとにshuffleを許容するとtrainとvalid分割間でも混同してしまうので、ここでは初回のみshuffleするだけに留めておく
            if self.use_validation == True:
                self.valid_data_tensor = self.data_tensor.take(self.nb_valid_sample)
                self.data_tensor = self.data_tensor.skip(self.nb_valid_sample) # train
                # self.valid_data_tensor = self.valid_data_tensor.shuffle(self.nb_valid_sample) # validはshuffleしない
                self.data_tensor = self.data_tensor.shuffle(self.nb_train_sample) # ここで各epごとでtrain内shuffleを許可する
                self.valid_data_tensor = self.valid_data_tensor.batch(self.batch_size) # validはこのときしか存在しないのでここでbatch作っておく
            else:
                pass
        self.data_tensor = self.data_tensor.batch(self.batch_size) # train shuffleなし/あり、testでもbatch化は必要なためここで共通して実行させる

        self.steps_per_epoch = self.nb_train_sample // self.batch_size \
                                if self.nb_train_sample % self.batch_size == 0 \
                                else self.nb_train_sample // self.batch_size + 1
        # while(True):
        #     for image in self.valid_data_tensor:
        #         print(tf.shape(image))
        #         image = self.postprocessing(image)
        #         print(image)
        #         plt.imshow(image[0])
        #         plt.savefig('./out.png')
        #         input()

    @tf.function
    def tf_random_rotate(self, image, input_size):
        # def random_rotate(image):
        #     image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
        #     return image
        def random_rotate(image, degree):
            # image : 32, 32, 3, eagertensor(値が出る)
            # image_np : 32, 32, 3, numpy array
            image_np=image.numpy()
            degree = degree.numpy()
            if tf.rank(image)==4: # [bs, h, w, c]
                X=Parallel(n_jobs=-1)([delayed(random_rotation)(an_image_np, degree, 0, 1, 2) for an_image_np in image_np])
                X=np.asarray(X) 
            elif tf.rank(image)==3: # [h, w, c]
                X=random_rotation(image_np, degree, 0, 1, 2)
            return X
        image = tf.py_function(random_rotate, [image, 90], [tf.float32]) # tf.Tensor : eager型 # shape消える # map内ではTensor型でgraphモードだがpy_function下では一時的にeagerがONになる
        image_x = image[0] # image_x : Tensor : Graph型
        image_x.set_shape(input_size)
        return image_x

    def preprocessing(self, filename): # bsを削減した形で書く
        string = tf.io.read_file(filename) # Tensor("ReadFile:0", shape=(), dtype=string)
        image = tf.image.decode_image(string) # decode_imageはshapeが出ないのでダメ -> set_shapeでshapeを指定してからresizeさせる
        image.set_shape(self.input_size) # tf.TensorShape([feature['input_size'], feature['input_size'], 1])
        image = tf.image.resize(image, [self.input_size[0], self.input_size[1]])
        image = self.tf_random_rotate(image, self.input_size) # (h, w, c) Tensor : graph型
        image = normalize(image, normalized_range=self.normalized_range)
        return image

    def postprocessing(self, tensor): # bsを削減した形で書く
        image = denormalize(tensor, normalized_range=self.normalized_range)
        image = tf.cast(image, tf.uint8)
        return image

def normalize(x, normalized_range='tanh'):
    if normalized_range == 'tanh':#画素値を-1~1に収める
        return (x / 255 - 0.5) / 0.5#画素値0~255 -> 0~1 -> -0.5~0.5 ->-1~1
    elif normalized_range == 'sigmoid':#画素値を0~1に収める
        return x / 255#画素値0~255 -> 0~1
    else:
        raise NotImplementedError

def denormalize(x, normalized_range='tanh'):
    if normalized_range == 'tanh':
        return ((x + 1.) / 2 * 255)
    elif normalized_range == 'sigmoid':
        return (x * 255)
    else:
        raise NotImplementedError

# WIP
def load_from_path(): # hierarchical search and create list
    pass



# ref
# https://www.tensorflow.org/tutorials/load_data/images#Performance
# https://www.tensorflow.org/guide/data_performance
# https://qiita.com/S-aiueo32/items/c7e86ef6c339dfb013ba
# https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=ja
