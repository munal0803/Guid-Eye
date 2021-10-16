# Tech-Eye
A python software that can help blind people find things like laptops, phones, etc the same way a guide dog guides a blind person in finding his way.
# The problem GuidEye solves
Consider a scenario. A blind person is alone at home and needs something like he wants to find his movable chair but has forgotten where it was placed. Normally he would have to guess the direction and try by reaching out to the distances if there is no one to help. In this case he would have to waste a lot of energy and time and there is also a risk of them hurting or wounding themselves in the process. Our project solves this problem.

A person using this would simply open this project, it would ask him what he wants and he would simply say "chair", the project would detect and start searching for the chair with the help of machine learning item detection using the camera of the device. The person then would have to simply rotate and would try to find the chair using his phone. As soon as the software detects a chair, it calls out just like a metal detector beeps. Following this sound, the person can find the object he wants (in this case a chair) and reach to it with ease.

Thus, this software acts like a guide dog helping out a blind person

# Challenges we ran into
Challenges we ran into while coming up and implementing this project :
1. We wanted to solve a problem for blind people but were confused how could we do it.
2. After coming up with this idea voice recognition and speaker acted as a small problem but was resolved.
3. Training a model for detecting posed a problem.
4. After this, coming up with a way of transferring and connecting a phone to the software

# Technologies used
 TensorFlow 
 OpenCV
 PythonSpeech 
 ** RecognitionText-to-Speech **
 SSD
