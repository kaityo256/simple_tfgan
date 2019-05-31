all: mnist.gif
PANDOC=pandoc -s --mathjax -t html --template=template

mnist.gif: mnist.tfrecord
	python gan_test.py

mnist.tfrecord:
	python mnist.py


.PHONY: clean gif

gif:
	convert -delay 10 -loop 0 gen*.png mnist.gif

web:
	$(PANDOC) README.md -o index.html

clean:
	rm -f gen*.png
	rm -f train-images-idx3-ubyte.gz
	rm -f mnist_train.tfrecord
