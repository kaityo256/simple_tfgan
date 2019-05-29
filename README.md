# Minimal sample code for TFGAN

## What is this?

[TensorFlow Models](https://github.com/tensorflow/models) is a quite nice tutorial. You can find a TFGAN example at [models/research/gan/](https://github.com/tensorflow/models/tree/master/research/gan). There is a Jupyter Notebook `tutorial.ipynb`. You can enjoy TFGAN just by executing cell by cell. Then you will obtain generated images of MNIST. ...Then you would be at a loss at what to do next. I did.

So I extract minimum codes from the tutorial which contains just two files with 270 lines. I hope it would help you to understand TFGAN and to go forward.

## How to use

Just run `make`. Or you can execute step by step.

```sh
python prepare_data.py
python gan_test.py
```

Then you will have generated images from `gen000.png` to `gen063.png`. It will take 10 - 15 minites.

Here are generated images.

![gen000.png](fig/gen000.png)
![gen005.png](fig/gen005.png)
![gen010.png](fig/gen010.png)
![gen063.png](fig/gen063.png)

You can see that the generated images becomes clearer and clearer as learning proceeds.

If you want, you can make an animation GIF file with ImageMagick as follow.

```sh
convert -delay 10 -loop 0 gen*.png mnist.gif
```

## License

The source codes in this repository are just extracted ones those from [TensorFlow Models](https://github.com/tensorflow/models). So the copyright holders are TensorFlow Authors. These modified files are available under Apache 2.0 License. See [LICENSE](LICENSE) for details.
