# Image Color Feature Extraction & Clustering

This project extracts dominant colors from an image using unsupervised clustering.

## Overview
Pixel-level color features are extracted from images, transformed into LAB color space, and clustered using K-Means.  
The number of clusters is automatically selected using silhouette score, and each color’s pixel proportion is reported.

## Tech Stack
- Python
- NumPy
- scikit-learn
- scikit-image
- Flask
