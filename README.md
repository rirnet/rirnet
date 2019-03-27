# rirnet

## Abstract 
This thesis work discuss the possibility to extract the impulse response function needed to construct a reverberant speech signal given that there exist an anechoic speech signal obscured by system influence, using deep learning. We propose two frameworks for combining deep autoencoder neural networks (for learning impulse response function features using point cloud representations) with convolutional neural networks that extract latent representations of the impulse response functions, either consuming time-series directly or a MFCC representation of the signal. We also present a framework that approximate the impulse response function from MFCC input directly. An emphasis is put on how to represent and obtain simulated data. Finally, we present results that show that the impulse response function can be extracted from reverberant signals with some accuracy.

<p align="center">
<img src ="https://raw.githubusercontent.com/rirnet/rirnet/master/tools/diagram.png" width="600" />
</p>
