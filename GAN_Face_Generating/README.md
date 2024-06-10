# Generating-Faces-with-DCGAN
Implementation of DCGAN on celebA dataset to generate faces

**Large-scale CelebFaces Attributes (celebA) dataset**
- CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.
* Data can be downloaded from here 
https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip

**Pre-processing and data loading**
* We don't need the annotations so we will have to crop images. Now, these are color images. Thus, depth is 3 (RGB — 3 color channels).
* We can resize & crop the images to a size of 32x32. This can be later converted into tensors.
* 
**Visualizing our training data**
* We will now generate a batch of data and visualize the same. Please note that the np.transpose function translates the image dimension by the order specified. For example, an RGB image of shape 3x32x32 gets transposed to 32x32x3 upon calling the following function: np.transpose(img,(1,2,0))
  ![gen images](https://github.com/abulzunayed/Machine-learning/assets/122612945/8da6399b-b1ed-4883-b0c8-85092d6aca06)

**Create Discriminator  and Generator Architecture**
  * We shall now define our discriminator network. The discriminator as we know is responsible for classifying the images as real or fake. Thus this is a typical classifier network.
  * The generator network is responsible for generating fake images that could fool the discriminator network into being classified as real. Over time the generator becomes pretty good in fooling the discriminator.
  * 
**Training phase & Loss curves**

![losses](https://github.com/abulzunayed/Machine-learning/assets/122612945/0b1f9d18-e806-4a12-8ff3-cb5ad499e677)

**Finally Sample generation and View image **
* It is important that we rescale the values back to the pixel range(0–255).
* And finally, we have our generated faces below.
* 
![train img](https://github.com/abulzunayed/Machine-learning/assets/122612945/88199682-fef1-4972-affe-dd17179ce044)

