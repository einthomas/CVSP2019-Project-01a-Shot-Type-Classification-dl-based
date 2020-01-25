# Shot type classification of historical video material using deep learning
Four shot types can be differentiated: Extreme long shots, long shots, medium shots and close-up shots.

## Project structure
* `Common` contains mostly utility functionality.
* `Develop` contains scripts for building, training and using a model for predictions.
* `Doc` contains HTML source code documentation.
* `shotTypeML_pkg` contains a script that handles image, video and folder inputs,
uses the scripts in `Develop` to predict shot types and returns (and stores)
CSV containing the results.

## Installation
Run `python setup.py install` or `pip install .` in the top-level folder
`CVSP2019-Project-01a-Shot-Type-Classification-dl-based` that contains `setup.py`.

## Usage
### Training the network
`python -m Develop.trainNetwork` starts the training process. Intermediate
results such as training accuracy and validation loss are printed continuously.
The number of training epochs as well as the expected image size and paths of
training and validation datasets are set in `config.yaml`.

### Testing the network
`python -m Develop.predict` predicts labels for the test dataset. The result
such as the test accuracy as well as precision and recall are printed. The path
of the test dataset is set in `config.yaml`.

### Production usage
`python -m shotTypeML_pkg.main [-h] [-i path] [-v path] [-o path]` predicts the
shot types of the provided images or videos. The result is returned and stored
(if a path is specified via `-o`) as CSV.

The command line arguments are:

* `-h` prints usage information.
* `-i path` specifies a path where an input image or a folder containing images is located.
* `-v path` specifies a path where an input video or a folder containing videos is located.
* `-o path` specifies where the resulting CSV file is stored, otherwise the result is just returned.

Example usage:

* `python -m shotTypeML_pkg.main -i "C:/Data" -o "C:/Output"` predicts the shot
types of all images in the folder `C:/Data/` and stores the resulting CSV under
`C:/Output`
* `python -m shotTypeML_pkg.main -v "C:/Data/video.mp4" -o "C:/Output"` predicts
the shot types of the frames of the provided video and stores the resulting CSV
under `C:/Output`
