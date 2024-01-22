# Weather images classification

This project is aimed to classify type of weather depicted in image.

To achieve this goal, two models were trained: custom CNN network and VGG16 network with weights pretrained on ImageNet dataset. 

Both models were then also trained using augemented data (rotation, zoom, shift, flip). 

## Data

There are four classes in dataset: cloudy, rain, shine and sunrise.

Dataset details: Ajayi, Gbeminiyi (2018), “Multi-class Weather Dataset for Image Classification”, Mendeley Data, V1, doi: 10.17632/4drtyfjtfy.1

Exemplary images:
||
|---|
|<img src="assets/images.png" width="450" height="450">|

Augmented images are stored in `augmented` folder.

Exemplary augmented images:
|   |   |   |   |
|---|---|---|---|
|![](augmented_images/aug_cloudy1.jpg)|![](augmented_images/aug_rain28.jpg)|![](augmented_images/aug_shine24.jpg)|![](augmented_images/aug_sunrise71.jpg)|

### Results

| Model | Accuracy | Confusion matrix |
| --- | --- | --- |
| VGG16 | 0.93 | <img src="assets/vgg16_confusion_matrix.png" width="250" height="250"> |
| VGG16 with augmented images | 0.92 | <img src="assets/vgg16_aug_confusion_matrix.png" width="250" height="250"> |
| CNN | 0.91 | <img src="assets/cnn_confusion_matrix.png" width="250" height="250"> |
| CNN with augmented images | 0.96 | <img src="assets/aug_confusion_matrix.png" width="250" height="250"> |

### How to run

Install required packages using `pip install -r requirements.txt`.
