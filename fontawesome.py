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

import random
import os
import sys

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format)
    }))


font = "./fa-solid-900.ttf"

if not os.path.exists(font):
    print("Could not find a font file %s." % font)
    print("Please down load from")
    print("https://fontawesome.com/")
    exit()

icons = [0xf0fc, 0xf0f3, 0xf02d, 0xf518, 0xf207,
         0xf1b9, 0xf5e4, 0xf1ae, 0xf64f, 0xf51e]

num_images = 10000
images = np.empty((num_images, 28, 28, 1), dtype=np.uint8)
for i in range(num_images):
    img = Image.new('L', (28, 28))
    draw = ImageDraw.Draw(img)
    draw.font = ImageFont.truetype(font, 24)
    x = random.randint(2, 4)
    y = random.randint(2, 4)
    text = chr(random.choice(icons))
    draw.text((x, y), text, (255))
    nim = np.array(img)
    nim = nim.reshape((28, 28, 1))
    images[i:] = nim
    sys.stdout.write('\r>> Generating image %d/%d' % (i + 1, num_images))
    sys.stdout.flush()
    #filename = "test%04d.png" % i
    # img.save(filename)


image = tf.placeholder(dtype=tf.uint8, shape=(28, 28, 1))
encoded_png = tf.image.encode_png(image)
with tf.Session('') as sess:
    with tf.python_io.TFRecordWriter("fontawesome.tfrecord") as tf_writer:
        for i in range(num_images):
            png_string = sess.run(encoded_png, feed_dict={image: images[i]})
            example = image_to_tfexample(png_string, 'png'.encode())
            tf_writer.write(example.SerializeToString())

print('Successfully generated fontawesome.tfrecord.')
