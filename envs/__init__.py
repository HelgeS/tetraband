from .classification import ImageClassificationEnv
from .detection import ObjectDetectionEnv

from gym.envs.registration import register

BASENAMES = {
    'classification': 'ImageClassificationEnv',
    'detection': 'ObjectDetectionEnv'
}

for scenario in ['basic', 'rotation', 'shear', 'hierarchical']:
    for dataset in ['cifar10', 'imagenet']:
        register(
            id='ImageClassificationEnv-{scenario}-{dataset}-v0'.format(scenario=scenario, dataset=dataset),
            entry_point='envs:ImageClassificationEnv',
            kwargs={
                'scenario': scenario,
                'dataset': dataset
            }
        )

for scenario in ['basic', 'rotation', 'shear', 'hierarchical']:
    dataset = 'coco'
    register(
        id='ObjectDetectionEnv-{scenario}-{dataset}-v0'.format(scenario=scenario, dataset=dataset),
        entry_point='envs:ObjectDetectionEnv',
        kwargs={
            'scenario': scenario,
            'dataset': dataset
        }
    )
