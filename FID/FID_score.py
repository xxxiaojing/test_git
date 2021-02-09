import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from numpy.random import shuffle
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from skimage.transform import resize
from tensorflow.keras.datasets import cifar10
from PIL import Image
import glob
import random


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def get_image_preprocessed(image_path):
  image = load_webp(image_path)
  #image = image[:, :, :] / 127.5 - 1.
  return image

def load_webp(img_path):
    im = Image.open(img_path)
    return numpy.asarray(im)

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(128, 128, 3))
# load cifar10 images

real_data_path = glob.glob("/home/wang-jing/tensorflow_v1/image_resizer/png/car_image_original/" + "*")
generated_data_path = glob.glob("/home/wang-jing/tensorflow_v1/HoloGAN_car_final_360/HoloGAN/samples_many_target_image_update3/" + "*")

random.seed(5)
random.shuffle(real_data_path)

batch_images1 = [get_image_preprocessed(batch_file) for batch_file in real_data_path[0:10000]]
images1 = numpy.asarray(batch_images1)
batch_images2 = [get_image_preprocessed(batch_file) for batch_file in generated_data_path[:]]
#batch_images2 = [get_image_preprocessed(batch_file) for batch_file in real_data_path[30000:40000]]
images2 = numpy.asarray(batch_images2)


"""
print(type(batch_images))
print(batch_images.shape)
quit()

(images1, _), (images2, _) = cifar10.load_data()
shuffle(images1)
images1 = images1[:10000]
images2 = images2[:10000]
"""
print('Loaded', images1.shape, images2.shape)

# convert integer to floating point values
images1 = images1.astype('float32')
images2 = images2.astype('float32')
# resize images
images1 = scale_images(images1, (128, 128, 3))
images2 = scale_images(images2, (128, 128, 3))
print('Scaled', images1.shape, images2.shape)
# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)
# calculate fid
fid = calculate_fid(model, images1, images2)
print('FID: %.3f' % fid)