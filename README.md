# Semi-Supervised Learning in Computer Vision
Semi-supervised learning methods for Computer Vision have been advancing quickly in the past few years. Current state-of-the-art methods are simplifying prior work in terms of architecture and loss function or introducing hybrid methods by blending different formulations.
## 1. Self-TrainingPermalink
In this semi-supervised formulation, a model is trained on labeled data and used to predict pseudo-labels for the unlabeled data. The model is then trained on both ground truth labels and pseudo-labels simultaneously.
![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/1b0f69d8-8366-4c79-8858-139cb2f653e0)

### a. Pseudo-label
Dong-Hyun Lee proposed a straightforward and efficient formulation called “Pseudo-label” in 2013.

The idea is to train a model simultaneously on a batch of labeled and unlabeled images. The model is trained on labeled images in the usual supervised manner with a cross-entropy loss. The same model is used to get predictions for a batch of unlabeled images and the maximum confidence class is used as the pseudo-label. Then, cross-entropy loss is calculated by comparing model predictions and the pseudo-label for the unlabeled images.
![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/9f2b32a8-897b-413d-bbb8-8d2bbe6b1109)
The total loss is a weighted sum of the labeled and unlabeled loss terms.
$L=L_(labeled)$
