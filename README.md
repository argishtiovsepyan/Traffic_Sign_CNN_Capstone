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

The preparation of the dataset began with reading annotations, where image and class information were extracted from a CSV file, followed by image augmentation using techniques like rescaling and zooming to enrich the training set. The images were then loaded and resized to a standard size, and further augmentations were applied to create diverse representations. Label transformation was done to one-hot encode the class labels for multi-class classification, allowing the system to recognize different traffic signs. Finally, all the augmented images and labels were converted to numpy arrays, making them ready for input into the deep learning model. 

<p align="center">
<img src= "images/presentation/datagen.jpg">
</p>
