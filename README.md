# Carla-Self-Driving-Car
## Overview
This was my first ever machine learning project! I attempted to use a Convolutional Neural Network to predict the steering direction (and angle to a poor degree) in which the 
vehicle should turn in. I have used the [Carla](https://carla.org/) simulator to create an environment for this project. This repo contains the files which I have written
completely myself. If you would like to use them, follow the Carla installation and add them to this directory ```PATH\TO\CARLA\CARLA_0.9.9.4\WindowsNoEditor\PythonAPI\examples```
(make sure to remove ```PATH\TO\CARLA``` and include your own path)

## Process<br>
Firstly, I created a python script in ```generate_data.py``` to gather training data. The training data was initially a 1080x720 RGBA image which I later normalized
to be a 50x50x1 greyscale image. I gathered around 40,000 frames of the car driving which I later used to train my CNN model.

```cnn.py``` contains the code which defines and saves the CNN model. In this file, I build the model using TensorFlow and Keras, as well as train it and save it.

Lastly, in ```self_drive.py``` is the code which controls the car and uses the saved model to make predictions based on the current input from the cars camera. The output from
the model is a **1-hot** array which corresponds to the cars steering direction and angle. 


## Conclusion
Unfortunately, I was unable to test my final results and would not be surpirsed that my model isn't working as well as others. I was unable to test my final self-driving vehicle
due to a memory shortage on my graphics card. My current graphics card cannot support running the Carla server and the model to make the predictions at the same time. Nevertheless,
this was my first exposure to CNN's and the machine learning process and I have learned alot from attempting this project. I'm sure I have made a ton of mistakes, but I look 
forwards to learning much more about machine learning and potentially even coming back to this project to fix them.
