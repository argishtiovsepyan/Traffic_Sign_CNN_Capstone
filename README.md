# **The Road Ahead:**
### Creating a CNN for Traffic Sign Detection and Classification

<p align="center">
<img src= "images/presentation/traffic_signs.jpg">
</p>

# **Introduction**
Tesla's automated driving cars have revolutionized the way we perceive traffic sign detection. It's a marvel that evokes the fascination I have with machine learning and its real world applications. Traffic sign detection is an integral part of autonomous driving technology that ensures safety and regulation compliance. My curiosity led me to explore how a CNN could be implemented to detect and interpret traffic signs, using cutting-edge technologies such as TensorFlow and Keras. I also made use of other essential libraries like Pandas, NumPy, Matplotlib and SciKit-Learn, creating a comprehensive toolkit for traffic sign detection. The journey led me to develop three Jupyter notebooks, each serving a unique purpose: Exploratory Data Analysis (EDA), Modeling, and Grad-CAM Visualization. The future of driving is here, and it's governed by the silent commands of traffic signs, interpreted by the brilliant minds of machines.

<p align="center">
<img src= "images/presentation/tf_keras.jpeg">
</p>

# **Finding Data and EDA**
Like all projects, the first step was to find data. Originally, I planned on using the full LISA (Laboratory for Intelligent and Safe Automobiles) traffic sign dataset containing 6,610 images and 47 classes. Unfortunately, the original source to this data from UC San Diego is no longer available. Luckily, on Kaggle I found a version of the dataset called Tiny LISA which contains 900 images and 9 classes. It also contained an annotations CSV where each annotation is a row containing the name of the file with enumeration and the traffic sign class label to which the file belongs. The images were 704 x 480 pixels in RGB format. 

The preparation of the dataset began with reading annotations, where image and class information were extracted from a CSV file, followed by image augmentation using techniques like rescaling and zooming to enrich the training set. The images were then loaded and resized to a standard size. Label transformation was done to one-hot encode the class labels for multi-class classification, allowing the system to recognize different traffic signs. Finally, all the augmented images and labels were converted to numpy arrays, making them ready for input into the deep learning model. 

<p align="center">
<img src= "images/presentation/datagen.jpg">
</p>

Using the generator, I acquired a total of 2700 images distributed across the same 9 classes. These were divided into three different sets: a training set consisting of 2160 images, a validation set with 270 images, and a hold-out set, also comprising 270 images.

# **Modeling**
Utilizing Keras's sequential model, I constructed the CNN and explored different architectural designs. After a meticulous process of fine-tuning various hyperparameters, it became evident that the ReLU activation function was optimal for this specific challenge. Adjustments were made to the number of epochs and batch size, though these modifications yielded minimal impact on the overall performance. In my examination of the two cold start models, I noted that the epochs tended to plateau quite early, a phenomenon addressed by implementing early stopping. Conversely, the transfer learning model employing VGG19 consistently reached the maximum number of epochs. Due to the VGG19 model's considerable slowness and computational expense with similar results, I ultimately chose to abandon that approach.

<table align="center" width="100%">
  <tr>
    <th align="center">Model Version 1</th>
    <th align="center">Model Version 2</th>
  </tr>
  <tr>
    <td align="center">
      <img src="images/presentation/model_v1_h5.svg" alt="Model Version 1" width="100%">
    </td>
    <td align="center">
      <img src="images/presentation/model_v2_h5.svg" alt="Model Version 2" width="100%">
    </td>
  </tr>
</table>
