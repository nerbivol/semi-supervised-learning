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

$L=L_{labeled}+\alpha_t * L_{unlabeled}$

### Noisy Student
Xie et al. proposed a semi-supervised method inspired by Knowledge Distillation called “Noisy Student” in 2019.

The key idea is to train two models called “Teacher” and “Student”. The teacher model is first trained on the labeled images and then used to infer the pseudo-labels for the unlabeled images. These pseudo-labels can either be soft-label or converted to hard-label by taking the most confident class. Then, the labeled and unlabeled images are combined and a student model is trained on this combined data. The images are augmented using RandAugment as a form of input noise. Also, model noise such as Dropout and Stochastic Depth are incorporated in the student model architecture.

![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/df268c4c-bd10-4b03-ad54-357b20d61f66)

Once a student model is trained, it becomes the new teacher, which is repeated for iterations.

## 2. Consistency Regularization
This paradigm uses the idea that model predictions on an unlabeled image should remain the same even after adding noise. We could use input noise such as Image Augmentation and Gaussian noise. Noise can also be incorporated in the architecture itself using Dropout.

![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/eef5a8c7-aa4b-42be-a75d-10e8b76f6113)

### a. π-model
This model was proposed by Laine et al. in a conference paper at ICLR 2017.

The key idea is to create two random augmentations of an image for both labeled and unlabeled data. Then, a model with dropout is used to predict the label of both these images. The square difference of these two predictions is used as a consistency loss. For labeled images, we also calculate the cross-entropy loss. The total loss is a weighted sum of these two loss terms. A weight w(t) is applied to decide how much the consistency loss contributes in the overall loss.

![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/ddb43220-570e-4732-8353-dd1dcaf8c91f)

### b. Temporal Ensembling
This method was also proposed by Laine et al. in the same paper as the pi-model. It modifies the π-model by leveraging the Exponential Moving Average(EMA) of predictions.

The key idea is to use the exponential moving average of past predictions as one view. To get another view, we augment the image as usual and a model with dropout is used to predict the label. The square difference of current prediction and EMA prediction is used as a consistency loss. For labeled images, we also calculate the cross-entropy loss. The final loss is a weighted sum of these two loss terms. A weight w(t) is applied to decide how much the consistency loss contributes in the overall loss.

















From source: [Amit Chaudhary. Semi-Supervised Learning in Computer Vision](https://amitness.com/2020/07/semi-supervised-learning/#a-mixmatch)
