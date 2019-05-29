all: mnist.gif

mnist.gif: mnist_train.tfrecord
	python gan_test.py

mnist_train.tfrecord:
	python prepare_data.py


.PHONY: clean gif

gif:
	convert -delay 10 -loop 0 gen*.png mnist.gif

clean:
	rm -f gen*.png
	rm -f train-images-idx3-ubyte.gz
	rm -f mnist_train.tfrecord
