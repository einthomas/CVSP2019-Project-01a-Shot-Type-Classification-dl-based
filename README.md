# Shot type classification of historical video material using deep learning
Four shot types can be differentiated: Extreme long shots, long shots, medium shots and close-up shots.

## Project structure
* `Demo` contains example data, a trained model and weights and a script that
shows how to use the installed package to classify the shot types of images and
videos.
* `Develop` contains scripts for training a model.
* `Doc` contains HTML source code documentation.
* `shotTypeML_pkg` contains an interface  that handles image, video and folder
inputs, predicts shot types and returns (and stores) CSV containing the results.

## Installation
Run `python setup.py install` in the top-level folder
`CVSP2019-Project-01a-Shot-Type-Classification-dl-based` that contains `setup.py`.

## Usage
### Training the network
`python Develop/trainNetwork.py -config Develop/config.yaml` starts the training
process. Intermediate results such as training accuracy and validation loss are
printed continuously. The number of training epochs as well as the expected
image size and paths of training and validation datasets are set in
`Develop/config.yaml`.

### Testing and evaluation
`python Develop/predict.py -config Develop/config.yaml` predicts labels for the
test image dataset specified in `Develop/config.yaml`. The result such as the
test accuracy as well as precision and recall are printed.

### Production usage
`demo.py` in `Demo` shows how to use the installed package to predict the shot
types of images and videos.
