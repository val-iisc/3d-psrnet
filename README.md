# 3D-PSRNet
This repository contains the source codes for the paper [3D-PSRNet: Part Segmented 3D Point Cloud Reconstruction From a Single Image](https://arxiv.org/abs/1810.00461).</br>
Accepted at *3D Reconstruction Meets Semantics* - ECCV 2018 Workshop

## Overview
Given a single-view image of an object, our network is able to predict its 3D point cloud, while also simultaneously computing a semantic segmentation. During training, a location-aware segmentation loss is used, which enables the integration of knowledge from both the predicted semantics and the reconstructed geometry. This form of joint optimization yields improved performance on both tasks.
![Overview of 3D-PSRNet](images/approach_overview.png)

## Sample Results
Below are a few sample reconstructions from our trained model.
![3D-PSRNet_sample_results](images/sample_results.png)

