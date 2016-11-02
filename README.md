# dcgan-tensorflow-classifier
Tensorflow implementation of DCGAN on the mnist dataset. Also includes code to train a classifier using the discriminator network.


## Dependencies

You will need the following Python packages:
```
numpy==1.11.2
Pillow==3.4.2
protobuf==3.0.0
scikit-learn==0.18
scipy==0.18.1
tensorflow==0.11.0rc0
```

## Usage

You can start training the GAN with `python main.py --logdir=logs`

Tensorboard will let you see samples `tensorboard --logdir=logs`

After some training, you may train the classifier `python main.py --logdir=logs --classifier`. The classifier is only updating weight of the last layer of the discriminator and trains on a reduced training set (around 400 images).

You can get images from the generator using `python main.py --logdir logs --sampledir samples`
