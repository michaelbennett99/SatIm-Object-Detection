# Object Detection In Aerial Images

This project uses the YOLOv8s model to detect objects in canonical satellite image datasets. The datasets used are [DOTA](https://captain-whu.github.io/DOTA/index.html), a large dataset of real aerial images collected from a variety of platforms, and [VALID](https://sites.google.com/view/valid-dataset), a dataset of synthetic aerial images.

The aim of the project was to evaluate the performance of state-of-the-art object detection models (that are trainable by an individual) on both these datasets, and to analyse the potential for transfer learning from cheap synthetic data to real data.

We found that the YOLOv8s model performed better on the synthetic dataset, with near perfect accuracy, but that MAP50 was significantly lower (at 0.56) on the real dataset. Moreover, pretraining on synthetic data was not helpful, as there was no difference in performance between zero pretraining and pretraining on the synthetic dataset when training and evaluating on DOTA. This suggest that the images in VALID are not similar enough to the real images featured in DOTA to allow for effective transfer learning.
