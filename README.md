# Tetraband - Test Transformation Bandit

`Tetraband` is an implementation of Adaptive Metamorphic Testing (AMT).
AMT extends traditional metamorphic testing to select one of the multiple metamorphic relations available for a program. 
By using contextual bandits, AMT learns which metamorphic relations are likely to transform a source test case, such that it has higher chance to discover faults.

More details are in the preprint of the paper [Adaptive Metamorphic Testing with Contextual Bandits](https://arxiv.org/abs/1910.00262).

## Running Tetraband

Example commandline call:
```
python run.py --environment classification --scenario hierarchical --dataset cifar10 --agent bandit --iterations 5000
```

List available options: `python run.py --help`

## Environments

We include OpenAI Gym environments, i.e. image classification and object detection (see code in [envs](envs/)).
Each of the environments has several options, which specify which dataset is used and which actions are available.

These environments can be used by importing the `envs` module and instantiating the wanted configuration:
```
import gym
import envs
env = gym.make('ImageClassificationEnv-basic-cifar10-v0')
print("Number of actions: ", env.action_space.n)
print("Observation space: ", env.observation_space)
print("Actions: ", env.action_names())
```

### Datasets
Available datasets (will be automatically downloaded):

- CIFAR-10 (image classification)
- Imagenet (image classification)
- MS COCO (object detection)

### Configurations

- `basic`: 9 MRs (Blur, Flip L/R, Flip U/D, Grayscale, Invert, Rotate(-30), Rotate(30), Shear(-20), Shear(20))
- `rotation`: Rotate(-90), Rotate(-85), Rotate(-80), ..., Rotate(80), Rotate(85), Rotate(90) (except Rotate(0))
- `shear`: Shear(-45), Shear(-40), Shear(-35), ..., Shear(35), Shear(40), Shear(45) (except Shear(0))
- `hierarchical`: A combination of `basic`, `rotation`, and `shear`.

## Requirements

Tetraband was developed and tested under Python 2.7 and might not work with Python 3.

Most required packages are installed via `pip -r requirements.txt`

Installing vowpal wabbit requires Boost.
If you are using anaconda, you can run `conda install boost` and `export BOOST_LIBRARYDIR=$CONDA_PREFIX/libs/` before installation.

For the object detection environments, you additionally need `pycocotools`, which can be installed either via

1) `conda install -c conda-forge pycocotools`
1) Cloning [the repository](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI) and running `make`.

### Object Detection API

The object detection environment is based on the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) from [tensorflow/models](https://github.com/tensorflow/models).

This is already setup, the links below are for reference.
[Installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

A good introduction is given in their [tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/oid_inference_and_evaluation.md).

In case of problems, ensure the object detection API is included in `$PYTHONPATH`.
But this is mostly handled by having the `object_detection` package copied into this project.

## License

[MIT License](LICENSE)
