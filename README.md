# plant-disease-detection
In this project, we built and trained a model which will be able to identify different plant-diseases and accordingly classify them. We used a kaggle dataset having images of 38 different plant-diseases. Our model was trained on Convolution Neural Network ( CNN ) architecture which helped us to achieve a training accuracy of 85.04%,validation accuracy of 87.72% and testing accuracy of 84.85%. In the training stage we experimented with different model architectures and stumbled upon some important findings regarding architectures. In short, we concluded that, in the training stage, along with bringing more dataset for better results, tuning model architecture should also be given a thought.
## Dependencies<br/>
In order to install the dependencies you can use the below command. You need to run this command from inside your cloned project directory.<br/>
```pip install requirements.txt```
## Parameters<br/>
``` epochs = 25```<br/>
```learning rate = 0.01```<br/>
```optimizer = adam ```<br/>
```loss = categorical_crossentropy ``` <br/>
``` patience = 3 ```<br/>
```model checkpoint period = 2 ```<br/>
```batch_size = 3 ```<br/>
## Convolution Neural Network ( CNN )
![CNN](CNN.jpg)<br/>
source - <b>towardsdatascience.com</b><br/><br/>
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.They use a mathematical operation called convolution in place of general matrix multiplication in at least one of their layers. Below is a gif showing how convolution works.<br/><br/>
source - <b>wikimedia.commons</b><br/><br/>
![Animation](Convolution_animation.gif)
## Dataset
<b>Link to Kaggle Dataset</b> - https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset <br/><br/>
<b>Original Authors</b> <br/>
https://cwur.org/<br/>
https://nces.ed.gov/programs/digest/d14/tables/dt14_605.20.asp<br/>
http://www.shanghairanking.com/<br/>
https://www.timeshighereducation.com/content/world-university-rankings
## Training 
After cloning the repository you can run the below command and get the training and validation results along with a model which will be able to classify plant-diseases.<br/>
```python plant_disease_classification.py --train "Path/to/training/dataset" --valid "path/to/validation/dataset"```
## Testing
Once you are ready with your testing data,move them into a folder and give the folder path in place of <b>path/to/testing/data</b> in the below testing command.<br/>
```python testing.py --test "path/to/testing/data"```
## ❤️ Found this project useful ? <br/>
If you found this project useful, then please consider giving it a ⭐ on Github and sharing it with your friends via social media. It really motivates us to do more.
