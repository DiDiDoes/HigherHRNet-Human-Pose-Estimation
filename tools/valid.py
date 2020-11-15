# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import tensorflow as tf
from tqdm import tqdm
import cv2
import numpy as np

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config

from dataset import make_test_dataloader
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from utils.utils import create_logger
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # load graph
    graph_path = 'models/EfficientDet-D0 + HigherHRNet.pb'
    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # build estimator
    tf.import_graph_def(graph_def, name='TfPoseEstimator')
    graph = tf.get_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    persistent_sess = tf.Session(graph=graph, config=config)

    tensor_image = graph.get_tensor_by_name('TfPoseEstimator/image:0')
    tensor_output = [graph.get_tensor_by_name('TfPoseEstimator/final_layers.0/BiasAdd:0'),
                     graph.get_tensor_by_name('TfPoseEstimator/final_layers.1/BiasAdd:0')]

    # build dataloader
    test_dataset = make_test_dataloader(cfg)

    parser = HeatmapParser(cfg)
    all_preds = []
    all_scores = []

    pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None
    for i, (image, annos) in enumerate(test_dataset):
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        final_heatmaps = None
        tags_list = []

        for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
            input_size = cfg.DATASET.INPUT_SIZE
            image_resized, center, scale = resize_align_multi_scale(
                image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
            )

            image_resized = image_resized / 255
            image_resized = image_resized.astype(np.float32)
            image_resized[:,:,0] = (image_resized[:,:,0] - 0.485) / 0.229
            image_resized[:,:,1] = (image_resized[:,:,1] - 0.456) / 0.224
            image_resized[:,:,2] = (image_resized[:,:,2] - 0.406) / 0.225

            _, heatmaps, tags = get_multi_stage_outputs(
                cfg, persistent_sess, tensor_output, tensor_image,
                image_resized, cfg.TEST.FLIP_TEST,
                cfg.TEST.PROJECT2IMAGE, base_size
                )

            final_heatmaps, tags_list = aggregate_results(
                cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

        final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
        tags = np.concatenate(tags_list, axis=4)

        grouped, scores = parser.parse(
            final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

        final_results = get_final_preds(
            grouped, center, scale,
            [final_heatmaps.shape[2], final_heatmaps.shape[1]]
        )

        if cfg.TEST.LOG_PROGRESS:
            pbar.update()

        all_preds.append(final_results)
        all_scores.append(scores)

    if cfg.TEST.LOG_PROGRESS:
        pbar.close()

    print(len(all_preds))
    name_values, _ = test_dataset.evaluate(
        cfg, all_preds, all_scores, './' #final_output_dir
    )

    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(logger, name_value, cfg.MODEL.NAME)
    else:
        _print_name_value(logger, name_values, cfg.MODEL.NAME)


if __name__ == '__main__':
    main()
