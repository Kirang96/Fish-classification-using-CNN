# Fish-classification-using-CNN

## THE PURPOSE

##### The primary goal of this mini-project is to demonstrate my knowledge in handling image data and building a baseline sequential model to make predictions within the constraints of time and resource power.

## RESOURCES AND TECHNOLOGIES USED

Data: 
- This project is done using 'A Large Scale Fish Dataset' created by Oğuzhan Ulucan available on Kaggle at https://www.kaggle.com/crowww/a-large-scale-fish-dataset.
- O.Ulucan, D.Karakaya, and M.Turkan.(2020) A large-scale dataset for fish segmentation and classification.
In Conf. Innovations Intell. Syst. Appli. (ASYU)
- This dataset contains 9 different seafood types collected from a supermarket in Izmir, Turkey
for a university-industry collaboration project at Izmir University of Economics, and this work
was published in ASYU 2020.
The dataset includes gilt head bream, red sea bream, sea bass, red mullet, horse mackerel,
black sea sprat, striped red mullet, trout, shrimp image samples.

![data_images](https://user-images.githubusercontent.com/29313141/126912309-450afe2d-c3fb-4106-bd37-89764f6e2c83.png)


It was initially run on kaggle notebook by directly importing data from kaggle. So you'll find the source of data as kaggle in the code.

- Tensorflow was used to build the model
 
## METHODOLOGY

- Image dataset is loaded using keras ImageDataGenerator. There are 9 classes of fishes to classify
- A sequential model is created with convolutional networks.
- Model is trained on the loaded dataset
- predictions are made on new fish images

## RESULTS

The model accuracy and loss was plotted against epoch to visualize the training. Accuracy went till 99%  to be platued.
![accuracy](https://user-images.githubusercontent.com/29313141/126912484-0d25aed0-b06f-4787-b45a-8ea44d62278f.png)
![loss](https://user-images.githubusercontent.com/29313141/126912487-13e6e1e3-2531-45b6-90aa-a3a6e765941d.png)

- If the accuracy was low, I could've augmented the data using ImageDataGenerator as well

## PERSONAL INFERENCES

- Learnt how to handle data using keras Datagenerator
- To build up a simple baseline CNN model for larger projects easily.

