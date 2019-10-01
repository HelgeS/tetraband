import numpy as np
import torch
from gym import spaces
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import networks
from envs.base import BaseEnv


class ImageClassificationEnv(BaseEnv):
    def __init__(self, scenario, evaluation='difference', dataset='cifar10', random_images=True):
        super(ImageClassificationEnv, self).__init__(scenario, evaluation, random_images)

        network_architecture = "resnet34" if dataset == 'cifar10' else "resnet50"
        self.model, self.input_size = networks.get_model(network_architecture, dataset)
        self.model.eval()
        self.model.to(networks.get_device())

        # TODO We could even use an unlabelled dataset and compare original and modified output
        if dataset == 'cifar10':
            self.dataset = datasets.CIFAR10(root='cifar10',
                                            train=False,
                                            download=True)
            self.pre_transformation = transforms.Compose([])
            obs_size = 32
        elif dataset == 'imagenet':
            self.dataset = datasets.ImageNet(root='imagenet',
                                             split='val',
                                             download=True)
            self.pre_transformation = transforms.Compose([])
            obs_size = 224

        self.num_distinct_images = len(self.dataset)

        self.model_transformation = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(obs_size, obs_size, 3),
                                            dtype=np.uint8)

    def run_all_actions(self, batch_size=8):
        """ For baseline purposes """
        original_image, label = self._get_image(self.cur_image_idx)
        original_input = self.model_transformation(original_image)

        mod_inputs = []
        action_ids = []

        for action_idx in range(len(self.actions)):
            if self.is_hierarchical_action(action_idx):
                for param_idx in range(len(self.actions[action_idx][1])):
                    modified_image = self.get_action(action_idx, param_idx)(image=original_image)
                    modified_input = self.model_transformation(modified_image)
                    mod_inputs.append(modified_input)
                    action_ids.append((action_idx, param_idx))
            else:
                modified_image = self.get_action(action_idx)(image=original_image)
                modified_input = self.model_transformation(modified_image)
                mod_inputs.append(modified_input)
                action_ids.append((action_idx, None))

        input = TensorDataset(torch.stack([original_input] + mod_inputs))
        loader = DataLoader(input, batch_size=batch_size)
        outputs = []

        for batch in loader:
            batch = batch[0].to(networks.get_device())
            output = self.model(batch)
            _, prediction = output.max(1)
            outputs.extend(prediction.cpu().tolist())

        outputs = np.array(outputs)
        pred_original = outputs[0]
        pred_modified = outputs[1:]

        original_correct = pred_original == label

        results = []

        for pred, (act_idx, param_idx) in zip(pred_modified, action_ids):
            evaluation_result = pred == pred_original

            r = self._reward(pred, pred_original, act_idx, param_idx)
            act_name, param_name = self.get_action_name(act_idx, param_idx)

            info = {
                'action': act_name,
                'parameter': param_name,
                'action_reward': r[0],
                'parameter_reward': r[1],
                'original': pred_original,
                'prediction': pred,
                'label': label,
                'success': bool(evaluation_result),
                'original_score': bool(original_correct),
                'modified_score': bool(pred == label)
            }
            results.append(info)

        return results

    def step(self, action):
        action_idx, parameter_idx = action
        # Apply transformation to current image
        original_image, label = self._get_image(self.cur_image_idx)
        modified_image = self.get_action(action_idx, parameter_idx)(image=original_image)

        # Input image into SUT
        original_input = self.model_transformation(original_image)
        modified_input = self.model_transformation(modified_image)

        input = torch.stack((modified_input, original_input))
        input = input.to(networks.get_device())

        output = self.model(input)
        _, prediction = output.max(1)
        pred_modified, pred_original = prediction.cpu().tolist()

        original_correct = pred_original == label
        modified_correct = pred_modified == label

        # Check result
        # The learning signal needs to be whether the transformation had an impact on the outcome
        # Whether the test failed is to be decided outside the environment.
        # In case the original output is already wrong, the expectation on the modified output might be different.
        # 0: No transformation impact
        # self.actions[action][2]: Transformation changed outcomes, action-dependent

        reward = self._reward(pred_modified, pred_original, action_idx, parameter_idx)
        observation = modified_image
        done = True
        info = {
            'original': pred_original,
            'prediction': pred_modified,
            'label': label,
            'original_score': original_correct,
            'modified_score': modified_correct
        }
        return observation, reward, done, info

    def _reward(self, pred_modified, pred_original, action_idx, parameter_idx=None):
        if pred_modified == pred_original:
            action_reward = 0
            parameter_reward = 0
        else:
            action_reward = self.actions[action_idx][2]

            if self.is_hierarchical_action(action_idx):
                parameter_reward = self.actions[action_idx][1][parameter_idx][2]
            else:
                parameter_reward = 0

        return (action_reward, parameter_reward)

    def _get_image(self, idx):
        image, label = self.dataset[idx]
        return image, label
