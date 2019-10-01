import os
import sys
import tarfile
import urllib

import numpy as np
import tensorflow as tf
# TODO Try to reference an external object_detection instead of keeping a copy
# Necessary because the tensorflow-models have a custom structure
# sys.path.append("../tensorflow-models/research/object_detection")
from PIL import Image
from gym import spaces

from envs.base import BaseEnv
from pycocotools.coco import COCO

sys.path.append('object_detection')
from object_detection.metrics import coco_tools

# sys.path.append("../tensorflow-models/research/slim")
# sys.path.append("../tensorflow-models/research/")
# from object_detection.utils import ops as utils_ops

# OBJECT DETECTION MODEL CONFIGURATION
# What model to download
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
# MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'models/' + MODEL_NAME + '/frozen_inference_graph.pb'


class ObjectDetectionEnv(BaseEnv):
    def __init__(self, scenario, evaluation='difference', dataset='coco', random_images=True):
        super(ObjectDetectionEnv, self).__init__(scenario, evaluation, random_images)

        self.model = get_model()
        self.model.as_default()
        self.tf_session = tf.Session(graph=self.model)

        # Dataset-specific
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(224, 224, 3),
                                            dtype=np.uint8)

        self._load_dataset('coco/')
        self.image_dir = 'coco/images/val2017/'
        self.categories = self.dataset.loadCats(self.dataset.getCatIds())

        self.map_difference = 0.05

        self.pre_transformation = lambda x: x
        self.model_transformation = prepare_image_as_input

    def __del__(self):
        self.tf_session.close()

    def _load_dataset(self, image_dir):
        self.bboxes = []

        ann_file = os.path.join(image_dir, 'annotations/instances_val2017.json')
        self.dataset = COCO(ann_file)
        self.num_distinct_images = len(self.dataset.getImgIds())

    def _initialize_indices(self):
        indices = sorted(self.dataset.getImgIds())

        if self.random_images:
            np.random.shuffle(indices)

        return indices

    def _get_image(self, idx):
        img_dict = self.dataset.loadImgs([idx])[0]
        image_path = os.path.join(self.image_dir, img_dict['file_name'])
        image = Image.open(image_path)
        ann_ids = self.dataset.getAnnIds([idx])
        annotations = self.dataset.loadAnns(ann_ids)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image, annotations

    def _query_model(self, inputs):
        output = run_inference(inputs, self.model, self.tf_session)
        return output

    def run_all_actions(self, batch_size=8):
        """ For baseline purposes """
        original_image, original_target = self._get_image(self.cur_image_idx)
        original_input = self.model_transformation(original_image)

        mod_inputs = []
        mod_targets = []
        action_ids = []

        for action_idx in range(len(self.actions)):
            if self.is_hierarchical_action(action_idx):
                for param_idx in range(len(self.actions[action_idx][1])):
                    modified_image, modified_target = self.get_action(action_idx, param_idx)(image=original_image,
                                                                                             bboxes=original_target)
                    modified_input = self.model_transformation(modified_image)
                    mod_inputs.append(modified_input)
                    mod_targets.append(modified_target)
                    action_ids.append((action_idx, param_idx))
            else:
                modified_image, modified_target = self.get_action(action_idx)(image=original_image,
                                                                              bboxes=original_target)
                modified_input = self.model_transformation(modified_image)
                mod_inputs.append(modified_input)
                mod_targets.append(modified_target)
                action_ids.append((action_idx, None))

        input = [original_input] + mod_inputs
        outputs = []

        for i in range((len(input) // batch_size) + 1):
            start = i * batch_size
            batch = input[start:start+batch_size]
            out = self._query_model(batch)
            outputs.extend(out)

        out_original = outputs[0]
        out_modified = outputs[1:]

        original_precision = self._evaluate_single(out_original, original_target)

        results = []

        for label, pred, (act_idx, param_idx) in zip(mod_targets, out_modified, action_ids):
            modified_precision = self._evaluate_single(pred, label)
            evaluation_result = modified_precision >= (original_precision - self.map_difference)

            r = self._reward(evaluation_result, act_idx, param_idx)
            act_name, param_name = self.get_action_name(act_idx, param_idx)

            info = {
                'action': act_name,
                'parameter': param_name,
                'action_reward': r[0],
                'parameter_reward': r[1],
                'original': out_original,
                'prediction': out_modified,
                'success': evaluation_result,
                'original_score': original_precision,
                'modified_score': modified_precision
            }
            results.append(info)

        return results

    def step(self, action):
        action_idx, parameter_idx = action
        # Apply transformation to current image
        original_image, original_target = self._get_image(self.cur_image_idx)
        modified_image, modified_target = self.get_action(action_idx, parameter_idx)(image=original_image,
                                                                                     bboxes=original_target)

        # Input image into SUT
        original_input = self.model_transformation(original_image)
        modified_input = self.model_transformation(modified_image)
        out_original, out_modified = self._query_model([original_input, modified_input])

        # Check result
        evaluation_result, modified_precision, original_precision = self._evaluate(out_original,
                                                                                   out_modified,
                                                                                   original_target,
                                                                                   modified_target)

        reward = self._reward(evaluation_result, action_idx, parameter_idx)
        observation = modified_image
        done = True
        info = {
            'original': out_original,
            'prediction': out_modified,
            'success': evaluation_result,
            'original_score': original_precision,
            'modified_score': modified_precision
        }
        return observation, reward, done, info

    def _evaluate(self, output_original, output_modified, label_original, label_modified):
        original_precision = self._evaluate_single(output_original, label_original)
        modified_precision = self._evaluate_single(output_modified, label_modified)

        return modified_precision >= (
                original_precision - self.map_difference), modified_precision, original_precision

    def _evaluate_single(self, detections_dict, groundtruth_list):
        groundtruth_dict = {
            'annotations': groundtruth_list,
            'images': [{'id': gt['image_id']} for gt in groundtruth_list],
            'categories': self.categories
        }

        if len(groundtruth_list) > 0:
            detections_list = coco_tools.ExportSingleImageDetectionBoxesToCoco(
                image_id=groundtruth_list[0]['image_id'],
                category_id_set=set([c['id'] for c in self.categories]),
                detection_boxes=detections_dict['detection_boxes'],
                detection_scores=detections_dict['detection_scores'],
                detection_classes=detections_dict['detection_classes']
            )
        else:
            detections_list = []

        # The COCO evaluation prints some information, which we don't care about
        with HiddenPrints():
            groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
            detections = groundtruth.LoadAnnotations(detections_list)
            evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections, iou_type='bbox')
            summary_metrics, _ = evaluator.ComputeMetrics()

        return summary_metrics['Precision/mAP']

    def _reward(self, evaluation_result, action_idx, parameter_idx=None):
        if evaluation_result:
            action_reward = 0
            parameter_reward = 0
        else:
            action_reward = self.actions[action_idx][2]

            if self.is_hierarchical_action(action_idx):
                parameter_reward = self.actions[action_idx][1][parameter_idx][2]
            else:
                parameter_reward = 0

        return action_reward, parameter_reward


# Helper methods to handle object detection networks
def get_model():
    if not os.path.isfile(PATH_TO_FROZEN_GRAPH):
        opener = urllib.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, 'models/')

        os.unlink(MODEL_FILE)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def run_inference(images, graph, sess):
    # with graph.as_default():
    #        with tf.Session() as sess:
    # Get handles to input and output tensors
    # ops = tf.get_default_graph().get_operations()
    ops = sess.graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    feed_dict = {}

    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ":0"
        if tensor_name in all_tensor_names:
            # tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            tensor_dict[key] = sess.graph.get_tensor_by_name(tensor_name)

    # image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    feed_dict[image_tensor] = np.stack(images)

    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict=feed_dict)

    outputs = []

    for idx in range(len(images)):
        # all outputs are float32 numpy arrays, so convert types as appropriate
        num_detections = int(output_dict['num_detections'][idx])

        det_dict = {
            'num_detections': num_detections,
            'detection_classes': output_dict['detection_classes'][idx][
                                 :num_detections].astype(np.uint8),
            'detection_boxes': output_dict['detection_boxes'][idx][:num_detections],
            'detection_scores': output_dict['detection_scores'][idx][:num_detections]
        }
        det_dict['detection_boxes'][:, 0] = det_dict['detection_boxes'][:, 0] * \
                                            images[idx].shape[0]  # Height
        det_dict['detection_boxes'][:, 1] = det_dict['detection_boxes'][:, 1] * \
                                            images[idx].shape[1]  # Width
        det_dict['detection_boxes'][:, 2] = det_dict['detection_boxes'][:, 2] * \
                                            images[idx].shape[0]  # Height
        det_dict['detection_boxes'][:, 3] = det_dict['detection_boxes'][:, 3] * \
                                            images[idx].shape[1]  # Width

        outputs.append(det_dict)

    return outputs


def load_image(path):
    image = Image.open(path).convert('RGB')
    return image


def prepare_image_as_input(image):
    im_width, im_height = image.size
    im_array = np.array(image.getdata())

    return im_array.reshape((im_height, im_width, 3)).astype(np.uint8)


class HiddenPrints(object):
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == '__main__':
    m = get_model()
    print(m)
    img_path = os.path.join('coco/images/val2017', os.listdir('coco/images/val2017')[5])
    print(img_path)
    img = load_image(img_path)
    x = run_inference([img], m, tf.Session())
    print(x)
