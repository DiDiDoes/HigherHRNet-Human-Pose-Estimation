# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(inputs, in_planes, out_planes, stride=1, name=None):
    """3x3 convolution with padding"""
    return layers.Conv2D(in_planes, kernel_size=3, strides=stride,
                         padding='same', use_bias=False, name=name)(inputs)


_BasicBlock_expansion = 1

def BasicBlock(inputs, inplanes, planes, stride=1, downsample=None, name=None):

    residual = inputs

    out = conv3x3(inputs, inplanes, planes, stride, name='%s.conv1' %name)
    out = layers.BatchNormalization(momentum=BN_MOMENTUM, name='%s.bn1' %name)(out)
    out = layers.ReLU()(out)

    out = conv3x3(out, inplanes, planes, stride, name='%s.conv2' %name)
    out = layers.BatchNormalization(momentum=BN_MOMENTUM, name='%s.bn2' %name)(out)

    if downsample is not None:
        residual = downsample(inputs)

    out = layers.Add()([out, residual])
    out = layers.ReLU()(out)

    return out


_Bottleneck_expansion = 4


def Bottleneck(inputs, inplanes, planes, stride=1, downsample=None, name=None):

    residual = inputs

    out = layers.Conv2D(planes, kernel_size=1, use_bias=False, name='%s.conv1' %name)(inputs)
    out = layers.BatchNormalization(momentum=BN_MOMENTUM, name='%s.bn1' %name)(out)
    out = layers.ReLU()(out)

    out = layers.Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=False, name='%s.conv2' %name)(out)
    out = layers.BatchNormalization(momentum=BN_MOMENTUM, name='%s.bn2' %name)(out)
    out = layers.ReLU()(out)

    out = layers.Conv2D(planes * _Bottleneck_expansion, kernel_size=1, use_bias=False, name='%s.conv3' %name)(out)
    out = layers.BatchNormalization(momentum=BN_MOMENTUM, name='%s.bn3' %name)(out)

    if downsample is not None:
        residual = downsample(inputs)

    out = layers.Add()([out, residual])
    out = layers.ReLU()(out)

    return out


def _check_branches(num_branches, blocks, num_blocks,
                    num_inchannels, num_channels):
    if num_branches != len(num_blocks):
        error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
            num_branches, len(num_blocks))
        logger.error(error_msg)
        raise ValueError(error_msg)

    if num_branches != len(num_channels):
        error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
            num_branches, len(num_channels))
        logger.error(error_msg)
        raise ValueError(error_msg)

    if num_branches != len(num_inchannels):
        error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
            num_branches, len(num_inchannels))
        logger.error(error_msg)
        raise ValueError(error_msg)


def _fuse_layer(inputs, num_inchannels, i, j, name=None):
    if j > i:
        output = layers.Conv2D(num_inchannels[i],
                               kernel_size=1,
                               strides=1,
                               padding='valid',
                               use_bias=False,
                               name='%s.fuse_layers.%d.%d.0' %(name, i, j))(inputs)
        output = layers.BatchNormalization(name='%s.fuse_layers.%d.%d.1' %(name, i, j))(output)
        output = layers.UpSampling2D(size=2**(j-i), interpolation='nearest')(output)
        return output
    elif j == i:
        return inputs
    else:
        output = inputs
        for k in range(i-j):
            if k == i - j - 1:
                num_outchannels_conv3x3 = num_inchannels[i]
                output = layers.Conv2D(num_outchannels_conv3x3,
                                       kernel_size=3, strides=2, padding='same', use_bias=False,
                                       name='%s.fuse_layers.%d.%d.%d.0' %(name, i, j, k))(output)
                output = layers.BatchNormalization(name='%s.fuse_layers.%d.%d.%d.1' %(name, i, j, k))(output)
            else:
                num_outchannels_conv3x3 = num_inchannels[j]
                output = layers.Conv2D(num_outchannels_conv3x3,
                                       kernel_size=3, strides=2, padding='same', use_bias=False,
                                       name='%s.fuse_layers.%d.%d.%d.0' %(name, i, j, k))(output)
                output = layers.BatchNormalization(name='%s.fuse_layers.%d.%d.%d.1' %(name, i, j, k))(output)
                output = layers.ReLU()(output)
        return output


def HighResolutionModule(inputs, num_branches, blocks, num_blocks, num_inchannels,
                         num_channels, fuse_method, multi_scale_output=True, name=None):

    _check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

    if num_branches == 1:
        raise Exception('Hey')
        #return [branches[0](inputs[0])]

    for i in range(num_branches):
        output = inputs[i]
        output = blocks(output, num_inchannels[i], num_channels[i], stride=1, name='%s.branches.%d.0' %(name, i))
        num_inchannels[i] = \
            num_channels[i] * _BasicBlock_expansion
        for j in range(1, num_blocks[i]):
            output = blocks(output, num_inchannels[i], num_channels[i], name='%s.branches.%d.%d' %(name, i, j))
        inputs[i] = output

    x_fuse = []
    for i in range(num_branches if multi_scale_output else 1):
        if i == 0:
            y = inputs[0]
        else:
            y = _fuse_layer(inputs[0], num_inchannels, i, 0, name)
        for j in range(1, num_branches):
            if i == j:
                y = layers.Add()([y, inputs[j]])
            else:
                y = layers.Add()([y, _fuse_layer(inputs[j], num_inchannels, i, j, name)])
        x_fuse.append(layers.ReLU()(y))

    return x_fuse, num_inchannels


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


def layer1_downsample(inputs):
    out = layers.Conv2D(64 * _Bottleneck_expansion, kernel_size=1, strides=1, use_bias=False, name='layer1.0.downsample.0')(inputs)
    out = layers.BatchNormalization(momentum=BN_MOMENTUM, name='layer1.0.downsample.1')(out)

    return out


def _transition_layer(inputs, num_channels_pre_layer, num_channels_cur_layer, name=None):
    num_branches_cur = len(num_channels_cur_layer)
    num_branches_pre = len(num_channels_pre_layer)

    outputs = []
    for i in range(num_branches_cur):
        if i < num_branches_pre:
            if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                output = layers.Conv2D(num_channels_cur_layer[i],
                                       kernel_size=3,
                                       strides=1,
                                       padding='same',
                                       use_bias=False,
                                       name='%s.%d.0' %(name, i))(inputs[i])
                output = layers.BatchNormalization(name='%s.%d.1' %(name, i))(output)
                outputs.append(layers.ReLU()(output))
            else:
                outputs.append(inputs[i])
        else:
            output = inputs[-1]
            for j in range(i+1-num_branches_pre):
                inchannels = num_channels_pre_layer[-1]
                outchannels = num_channels_cur_layer[i] \
                    if j == i-num_branches_pre else inchannels
                output = layers.Conv2D(outchannels, kernel_size=3, strides=2, padding='same', use_bias=False,
                                       name='%s.%d.%d.0' %(name, i, j))(output)
                output = layers.BatchNormalization(name='%s.%d.%d.1' %(name, i, j))(output)
            outputs.append(layers.ReLU()(output))

    return outputs


def _stage(inputs, layer_config, num_inchannels,
           multi_scale_output=True, name=None):

    num_modules = layer_config['NUM_MODULES']
    num_branches = layer_config['NUM_BRANCHES']
    num_blocks = layer_config['NUM_BLOCKS']
    num_channels = layer_config['NUM_CHANNELS']
    block = blocks_dict[layer_config['BLOCK']]
    fuse_method = layer_config['FUSE_METHOD']

    outputs = inputs
    for i in range(num_modules):
        # multi_scale_output is only used last module
        if not multi_scale_output and i == num_modules - 1:
            reset_multi_scale_output = False
        else:
            reset_multi_scale_output = True

        outputs, num_inchannels = HighResolutionModule(
                      outputs,
                      num_branches,
                      block,
                      num_blocks,
                      num_inchannels,
                      num_channels,
                      fuse_method,
                      reset_multi_scale_output,
                      name='%s.%d' %(name, i))

    return outputs, num_inchannels


def _make_final_layers(cfg, input_channels):
    dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
    extra = cfg.MODEL.EXTRA

    final_layers = []
    output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
        if cfg.LOSS.WITH_AE_LOSS[0] else cfg.MODEL.NUM_JOINTS
    final_layers.append([layers.Conv2D(
        filters=output_channels,
        kernel_size=extra.FINAL_CONV_KERNEL,
        strides=1,
        padding='same' if extra.FINAL_CONV_KERNEL == 3 else 'valid',
        name = 'final_layers.0'
    )])

    deconv_cfg = extra.DECONV
    for i in range(deconv_cfg.NUM_DECONVS):
        input_channels = deconv_cfg.NUM_CHANNELS[i]
        output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
            if cfg.LOSS.WITH_AE_LOSS[i+1] else cfg.MODEL.NUM_JOINTS
        final_layers.append([layers.Conv2D(
            filters=output_channels,
            kernel_size=extra.FINAL_CONV_KERNEL,
            strides=1,
            padding='same' if extra.FINAL_CONV_KERNEL == 3 else 'valid',
            name = 'final_layers.%d' %(i+1)
        )])

    return final_layers


def PoseHigherResolutionNet(inputs, cfg, **kwargs):

    inplanes = 64
    extra = cfg.MODEL.EXTRA

    # stem net
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name='bn1')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv2')(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name='bn2')(x)
    x = layers.ReLU()(x)

    # layer1
    downsample = None
    if inplanes != 64 * _Bottleneck_expansion:
        downsample = layer1_downsample

    x = Bottleneck(x, inplanes, 64, 1, downsample, name='layer1.0')
    inplanes = 64 * _Bottleneck_expansion
    for i in range(1, 4):
        x = Bottleneck(x, inplanes, 64, name='layer1.%d' %i)

    # stage2
    stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
    num_channels = stage2_cfg['NUM_CHANNELS']
    block = blocks_dict[stage2_cfg['BLOCK']]
    num_channels = [
        num_channels[i] * _BasicBlock_expansion for i in range(len(num_channels))
    ]

    x_list = _transition_layer([x], [256], num_channels, name='transition1')
    y_list, pre_stage_channels = _stage(x_list, stage2_cfg, num_channels, name='stage2')

    # stage3
    stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
    num_channels = stage3_cfg['NUM_CHANNELS']
    block = blocks_dict[stage3_cfg['BLOCK']]
    num_channels = [
        num_channels[i] * _BasicBlock_expansion for i in range(len(num_channels))
    ]

    x_list = _transition_layer(y_list, pre_stage_channels, num_channels, name='transition2')
    y_list, pre_stage_channels = _stage(x_list, stage3_cfg, num_channels, name='stage3')

    # stage4
    stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
    num_channels = stage4_cfg['NUM_CHANNELS']
    block = blocks_dict[stage4_cfg['BLOCK']]
    num_channels = [
        num_channels[i] * _BasicBlock_expansion for i in range(len(num_channels))
    ]

    x_list = _transition_layer(y_list, pre_stage_channels, num_channels, name='transition3')
    y_list, pre_stage_channels = _stage(x_list, stage4_cfg, num_channels, multi_scale_output=False, name='stage4')

    # final layers
    extra = cfg.MODEL.EXTRA
    final_layers = _make_final_layers(cfg, pre_stage_channels[0])
    
    input_channels = pre_stage_channels[0]
    dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
    deconv_cfg = extra.DECONV

    num_deconvs = extra.DECONV.NUM_DECONVS
    concat_layers = [layers.Concatenate() for i in range(num_deconvs)]
    deconv_config = cfg.MODEL.EXTRA.DECONV
    loss_config = cfg.LOSS

    pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    final_outputs = []
    x = y_list[0]
    y = x
    for l in final_layers[0]:
        y = l(y)
    final_outputs.append(y)

    for i in range(num_deconvs):
        if deconv_cfg.CAT_OUTPUT[i]:
            x = concat_layers[i]([x, y])
        #x = deconv_layers[i](x)
        if deconv_cfg.CAT_OUTPUT[i]:
            final_output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
                if cfg.LOSS.WITH_AE_LOSS[i] else cfg.MODEL.NUM_JOINTS
            input_channels += final_output_channels
        output_channels = deconv_cfg.NUM_CHANNELS[i]

        _layers = [
            layers.Conv2DTranspose(
                filters=output_channels,
                kernel_size=deconv_cfg.KERNEL_SIZE[i],
                strides=2,
                padding='same',
                use_bias=False,
                name = 'deconv_layers.%d.0.0' %i),
            layers.BatchNormalization(momentum=BN_MOMENTUM, name = 'deconv_layers.%d.0.1' %i),
            layers.ReLU()
        ]
        for l in _layers:
            x = l(x)
        for _ in range(cfg.MODEL.EXTRA.DECONV.NUM_BASIC_BLOCKS):
            x = BasicBlock(x, output_channels, output_channels, name = 'deconv_layers.%d.%d.0' %(i, _+1))
        input_channels = output_channels

        y = x
        for l in final_layers[i+1]:
            y = l(y)
        final_outputs.append(y)

    return final_outputs


def get_pose_net(cfg, is_train, **kwargs):
    inputs = keras.Input((384, 384, 3), name='image')
    outputs = PoseHigherResolutionNet(inputs, cfg, **kwargs)
    model = keras.Model(inputs, outputs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model
