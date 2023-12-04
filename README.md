# Semi-Supervised Learning in Computer Vision
Semi-supervised learning methods for Computer Vision have been advancing quickly in the past few years. Current state-of-the-art methods are simplifying prior work in terms of architecture and loss function or introducing hybrid methods by blending different formulations.
## 1. Self-Training
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

The key idea is to use the exponential moving average of past predictions as one view. To get another view, we augment the image as usual and a model with dropout is used to predict the label. The square difference between the current prediction and the EMA prediction is used as a consistency loss. For labeled images, we also calculate the cross-entropy loss. The final loss is a weighted sum of these two loss terms. A weight w(t) is applied to decide how much the consistency loss contributes to the overall loss.

### c. Mean Teacher
This method was proposed by Tarvainen et al.. The general approach is similar to Temporal Ensembling but it uses Exponential Moving Averages(EMA) of the model parameters instead of predictions.

The key idea is to have two models called “Student” and “Teacher”. The student model is a regular model with dropout. The teacher model has the same architecture as the student model but its weights are set using an exponential moving average of the weights of the student model. For a labeled or unlabeled image, we create two random augmented versions of the image. Then, the student model is used to predict label distribution for the first image. And, the teacher model is used to predict the label distribution for the second augmented image. The square difference between these two predictions is used as a consistency loss. For labeled images, we also calculate the cross-entropy loss. The final loss is a weighted sum of these two loss terms. A weight w(t) is applied to decide how much the consistency loss contributes to the overall loss.

![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/f365629a-4eb1-4e7e-90b8-d488f1aa3c4f)

### d. Virtual Adversarial Training
This method was proposed by Miyato et al.. It uses the concept of adversarial attack for consistency regularization.

The key idea is to generate an adversarial transformation of an image that will change the model prediction. To do so, first, an image is taken and an adversarial variant of it is created such that the KL-divergence between the model output for the original image and the adversarial image is maximized.

Then we proceed with the previous methods. We take a labeled/unlabeled image as the first view and take its adversarial example generated in the previous step as the second view. Then, the same model is used to predict label distributions for both images. The KL divergence of these two predictions is used as a consistency loss. For labeled images, we also calculate the cross-entropy loss. The final loss is a weighted sum of these two loss terms. A weight α is applied to decide how much the consistency loss contributes to the overall loss.

![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/864030db-3b15-4843-8dae-0bdd3023c29b)

### e. Unsupervised Data Augmentation
This method was proposed by Xie et al. and works for both images and text. Here, we will understand the method in the context of images.

The key idea is to create an augmented version of an unlabeled image using AutoAugment. Then, the same model is used to predict the label of both these images. The KL-divergence of these two predictions is used as a consistency loss. For labeled images, we only calculate the cross-entropy loss and don’t calculate any consistency loss. The final loss is a weighted sum of these two loss terms. A weight w(t) is applied to decide how much the consistency loss contributes to the overall loss.

![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/0de9907d-ffc8-4a03-87ab-a5920889792e)

## 3. Hybrid Methods
This paradigm combines ideas from previous work such as self-training and consistency regularization along with additional components for performance improvement.

### a. MixMatchPermalink
This holistic method was proposed by Berthelot et al..

To understand this method, let’s take a walk through each of the steps.
- I. For the labeled image, we create an augmentation of it. For the unlabeled image, we create K augmentations and get the model predictions on all K-images. Then, the predictions are averaged and temperature scaling is applied to get a final pseudo-label. This pseudo-label will be used for all the K-augmentations.

![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/8fdd3548-0a0c-4d72-8ced-6a4fb543c651)

- II. The batches of augmented labeled and unlabeled images are combined and the whole group is shuffled. Then, the first N images of this group are taken as $W_L$, and the remaining M images are taken as $W_U$.
  
![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/aff1fb61-3743-417c-91a1-1824b1504502)

- III. Now, Mixup is applied between the augmented labeled batch and group $W_L$. Similarly, mixup is applied between the M augmented unlabeled group and the $W_U$ group. Thus, we get the final labeled and unlabeled group. 

![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/dc0c9839-160e-49b2-8e84-956c25f5d656)

- IV. Now, for the labeled group, we take model predictions and compute cross-entropy loss with the ground truth mixup labels. Similarly, for the unlabeled group, we compute model predictions and compute mean square error(MSE) loss with the mixup pseudo labels. A weighted sum is taken of these two terms with λ weighting the MSE loss.

![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/d36e10d3-fde0-4ee5-bc33-3c04e87ee03f)

### b. FixMatch
This method was proposed by Sohn et al. and combines pseudo-labeling and consistency regularization while vastly simplifying the overall method. It got state-of-the-art results on a wide range of benchmarks.

As seen, we train a supervised model on our labeled images with cross-entropy loss. For each unlabeled image, weak augmentation and strong augmentations are applied to get two images. The weakly augmented image is passed to our model and we get prediction over classes. The probability for the most confident class is compared to a threshold. If it is above the threshold, then we take that class as the ground label i.e. pseudo-label. Then, the strongly augmented image is passed through our model to get a prediction over classes. This prediction is compared to the ground truth pseudo-label using cross-entropy loss. Both the losses are combined and the model is optimized.

![image](https://github.com/nerbivol/semi-supervised-learning/assets/68056715/8e5baec3-bf01-45f8-a767-f3cfb10a591b)

From source: Amit Chaudhary. [Semi-Supervised Learning in Computer Vision](https://amitness.com/2020/07/semi-supervised-learning/#a-mixmatch)

@misc{chaudhary2020semisupervised,
  title   = {Semi-Supervised Learning in Computer Vision},
  author  = {Amit Chaudhary},
  year    = 2020,
  note    = {\url{https://amitness.com/2020/07/semi-supervised-learning/}}
}
