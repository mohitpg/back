# Image Caption

This repository contains a production grade machine learning model served by tensorflow-serving and flask which performs image captioning with a front end interface designed with React.

# Key Features
<ul>
<li>User-friendly web interface for uploading images.</li>
<li>Backend powered by Flask, enabling seamless communication with the deep learning model.</li>
<li>The deep learning model is built using Keras, leveraging pre-trained convolutional neural networks (CNN) and recurrent neural networks (RNN) for image feature extraction and caption generation, respectively.</li>
<li>Dockerized the entire application for installability</li>
<li>Real-time image processing and caption generation.</li>
<li>Easy-to-understand UI built with React, allowing a smooth user experience.</li>
</ul>

# Installation
1. Run  `git clone https://github.com/mohitpg/ImageCaption.git` <br>
2. Go to the folder ImageCaption. <br>
3. Run `docker-compose up`. <br>
4. The application can be accessed at localhost:5000

# Acknowledgements
The model architecture and training code were adapted from <a href='https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8'> Harshall Lamba's Article </a>

# Screenshots
<div align="center">
 <img src='https://github.com/mohitpg/back/blob/main/frontend/public/ss1.png?raw=true'>
 
 <img src='https://github.com/mohitpg/back/blob/main/frontend/public/ss2.png?raw=true'>
 
 <img src='https://github.com/mohitpg/back/blob/main/frontend/public/ss3.png?raw=true'>

</div>




