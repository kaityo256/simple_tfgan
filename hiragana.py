# Copyright 2019 H. Watanabe All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format)
    }))


font = "./ipaexg.ttf"

if not os.path.exists(font):
    print("Could not find a font file ipaexg.ttf.")
    print("Please down load from")
    print("https://ipafont.ipa.go.jp/old/ipafont/download.html")
    exit()

num_images = 10000
images = np.empty((num_images, 28, 28, 1), dtype=np.uint8)
for i in range(num_images):
    img = Image.new('L', (28, 28))
    draw = ImageDraw.Draw(img)
    draw.font = ImageFont.truetype(font, 28)
    x = random.randint(-2, 2)
    y = random.randint(-2, 2)
    theta = random.randint(-10, 10)
    kana = chr(random.randint(ord('あ'), ord('ん')))
    draw.text((x, y), kana, (255))
    img = img.rotate(theta)
    nim = np.array(img)
    nim = nim.reshape((28, 28, 1))
    images[i:] = nim
    sys.stdout.write('\r>> Generating image %d/%d' % (i + 1, num_images))
    sys.stdout.flush()
    #filename = "test%04d.png" % i
    # img.save(filename)

print()

image = tf.placeholder(dtype=tf.uint8, shape=(28, 28, 1))
encoded_png = tf.image.encode_png(image)
with tf.Session('') as sess:
    with tf.python_io.TFRecordWriter("hiragana.tfrecord") as tf_writer:
        for i in range(num_images):
            png_string = sess.run(encoded_png, feed_dict={image: images[i]})
            example = image_to_tfexample(png_string, 'png'.encode())
            tf_writer.write(example.SerializeToString())

print('Successfully generated hiragana.tfrecord.')
