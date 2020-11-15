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
import numpy as np
import re
import collections
import math

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

#from .backbone import EfficientDetBackbone

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def Swish(x):
    return x * keras.activations.sigmoid(x)


def MBConvBlock(inputs, block_args, global_params, i):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """
    print(i)
    _block_args = block_args
    _bn_mom = global_params.batch_norm_momentum
    _bn_eps = global_params.batch_norm_epsilon
    has_se = (_block_args.se_ratio is not None) and (0 < _block_args.se_ratio <= 1)
    id_skip = block_args.id_skip  # skip connection and drop connect

    # Expansion phase
    inp = _block_args.input_filters  # number of input channels
    oup = _block_args.input_filters * _block_args.expand_ratio  # number of output channels

    # Depthwise convolution phase
    k = _block_args.kernel_size
    s = _block_args.stride

    # Squeeze and Excitation layer, if desired
    if has_se:
        num_squeezed_channels = max(1, int(_block_args.input_filters * _block_args.se_ratio))

    # Output phase
    final_oup = _block_args.output_filters

    # Expansion and Depthwise Convolution
    x = inputs
    if _block_args.expand_ratio != 1:
        x = layers.Conv2D(filters=oup, padding='same', kernel_size=1, use_bias=False, name='backbone_neck.backbone_net.model._blocks.%d._expand_conv.conv' %i)(inputs)
        x = layers.BatchNormalization(momentum=_bn_mom, epsilon=_bn_eps, name='backbone_neck.backbone_net.model._blocks.%d._bn0' %i)(x)
        x = Swish(x)

    x = layers.DepthwiseConv2D(
        padding='same', #filters=oup, groups=oup,  # groups makes it depthwise
        kernel_size=k, strides=s, use_bias=False, name='backbone_neck.backbone_net.model._blocks.%d._depthwise_conv.conv' %i)(x)
    x = layers.BatchNormalization(momentum=_bn_mom, epsilon=_bn_eps, name='backbone_neck.backbone_net.model._blocks.%d._bn1' %i)(x)
    x = Swish(x)

    # Squeeze and Excitation
    if has_se:
        x_squeezed = K.mean(x, [1, 2], keepdims=True)
        x_squeezed = layers.Conv2D(filters=num_squeezed_channels, padding='same', kernel_size=1, name='backbone_neck.backbone_net.model._blocks.%d._se_reduce.conv' %i)(x_squeezed)
        x_squeezed = Swish(x_squeezed)
        x_squeezed = layers.Conv2D(filters=oup, padding='same', kernel_size=1, name='backbone_neck.backbone_net.model._blocks.%d._se_expand.conv' %i)(x_squeezed)
        x = keras.activations.sigmoid(x_squeezed) * x

    x = layers.Conv2D(filters=final_oup, padding='same', kernel_size=1, use_bias=False, name='backbone_neck.backbone_net.model._blocks.%d._project_conv.conv' %i)(x)
    x = layers.BatchNormalization(momentum=_bn_mom, epsilon=_bn_eps, name='backbone_neck.backbone_net.model._blocks.%d._bn2' %i)(x)

    # Skip connection and drop connect
    input_filters, output_filters = _block_args.input_filters, _block_args.output_filters
    if id_skip and _block_args.stride == 1 and input_filters == output_filters:
        '''
        if drop_connect_rate:
            print(self.training)
            x = drop_connect(x, p=drop_connect_rate, training=self.training)
        '''
        x = layers.Add()([x, inputs])  # skip connection
    return x


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 384, 0.2),
        'efficientnet-b1': (1.0, 1.1, 384, 0.2),
        'efficientnet-b2': (1.1, 1.2, 384, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s'][0]))

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def EfficientNet(inputs, compound_coef, override_params=None):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """
    model_name = f'efficientnet-b{compound_coef}'
    blocks_args, global_params = get_model_params(model_name, override_params)
    assert isinstance(blocks_args, list), 'blocks_args should be a list'
    assert len(blocks_args) > 0, 'block args must be greater than 0'
    _global_params = global_params
    _blocks_args = blocks_args

    # Get static or dynamic convolution depending on image size
    #Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

    # Batch norm parameters
    bn_mom = _global_params.batch_norm_momentum
    bn_eps = _global_params.batch_norm_epsilon

    # Stem
    in_channels = 3  # rgb
    out_channels = round_filters(32, _global_params)  # number of output channels

    # Build blocks
    _blocks = []
    for block_args in _blocks_args:

        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters, _global_params),
            output_filters=round_filters(block_args.output_filters, _global_params),
            num_repeat=round_repeats(block_args.num_repeat, _global_params)
        )

        # The first block needs to take care of stride and filter size increase.
        _blocks.append([block_args, _global_params])
        if block_args.num_repeat > 1:
            block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
        for _ in range(block_args.num_repeat - 1):
            _blocks.append([block_args, _global_params])

    # Head
    in_channels = block_args.output_filters  # output of final block

    # Stem
    x = layers.Conv2D(out_channels, padding='same', kernel_size=3, strides=2, use_bias=False, name='backbone_neck.backbone_net.model._conv_stem.conv')(inputs)
    x = layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps, name='backbone_neck.backbone_net.model._bn0')(x)
    x = Swish(x)

    # Blocks
    feature_maps = []
    last_x = None
    for idx, block in enumerate(_blocks):
        x = MBConvBlock(x, block[0], block[1], idx)
        if last_x is not None and x.shape[1] < last_x.shape[1]:
            feature_maps.append(last_x)
        elif idx == len(_blocks) - 1:
            feature_maps.append(x)
        last_x = x

    return feature_maps[2:]


def SeparableConvBlock(inputs, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False, name=None):
    """
    created by Zylo117
    """
    if out_channels is None:
        out_channels = in_channels

    # Q: whether separate conv
    #  share bias between depthwise_conv and pointwise_conv
    #  or just pointwise_conv apply bias.
    # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

    depthwise_conv = layers.DepthwiseConv2D(padding='same', # in_channels, groups=in_channels,
                                            kernel_size=3, strides=1,  use_bias=False,
                                            name='%s.depthwise_conv.conv' %name)
    pointwise_conv = layers.Conv2D(out_channels, padding='same', kernel_size=1, strides=1, name='%s.pointwise_conv.conv' %name)

    if norm:
        # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
        bn = layers.BatchNormalization(momentum=0.01, epsilon=1e-3, name='%s.bn' %name)

    x = depthwise_conv(inputs)
    x = pointwise_conv(x)

    if norm:
        x = bn(x)

    if activation:
        x = Swish(x)

    return x


def BiFPN(inputs, num_channels, conv_channels, time=None, epsilon=1e-4, onnx_export=False, attention=True,
          use_p8=False, last=False):
    """
    modified by Zylo117
    """
    """

    Args:
        num_channels:
        conv_channels:
        first_time: whether the input comes directly from the efficientnet,
                    if True, downchannel it first, and downsample P5 to generate P6 then P7
        epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
        onnx_export: if True, use Swish instead of MemoryEfficientSwish
    """
    if use_p8:
        raise Exception('no p8 please.')

    """
    illustration of a minimal bifpn unit
        P7_0 -------------------------> P7_2 -------->
           |-------------|                ↑
                      ↓                |
        P6_0 ---------> P6_1 ---------> P6_2 -------->
           |-------------|--------------↑ ↑
                      ↓                |
        P5_0 ---------> P5_1 ---------> P5_2 -------->
           |-------------|--------------↑ ↑
                      ↓                |
        P4_0 ---------> P4_1 ---------> P4_2 -------->
           |-------------|--------------↑ ↑
                         |--------------↓ |
        P3_0 -------------------------> P3_2 -------->
    """

    # downsample channels using same-padding conv2d to target phase's if not the same
    # judge: same phase as target,
    # if same, pass;
    # elif earlier phase, downsample to target phase's by pooling
    # elif later phase, upsample to target phase's by nearest interpolation

    if attention:
        if time==0:
            p3 = inputs[0]
            p4 = inputs[1]
            p5 = inputs[2]

            p6_in = layers.Conv2D(num_channels, 1, padding='same', name='backbone_neck.bifpn.%d.p5_to_p6.0.conv' %time)(p5)
            p6_in = layers.BatchNormalization(momentum=0.01, epsilon=1e-3, name='backbone_neck.bifpn.%d.p5_to_p6.1' %time)(p6_in)
            p6_in = layers.MaxPooling2D(3, 2, padding='same')(p6_in)

            p7_in = layers.MaxPooling2D(3, 2, padding='same')(p6_in)

            p3_in = layers.Conv2D(num_channels, 1, padding='same', name='backbone_neck.bifpn.%d.p3_down_channel.0.conv' %time)(p3)
            p3_in = layers.BatchNormalization(momentum=0.01, epsilon=1e-3, name='backbone_neck.bifpn.%d.p3_down_channel.1' %time)(p3_in)

            p4_in = layers.Conv2D(num_channels, 1, padding='same', name='backbone_neck.bifpn.%d.p4_down_channel.0.conv' %time)(p4)
            p4_in = layers.BatchNormalization(momentum=0.01, epsilon=1e-3, name='backbone_neck.bifpn.%d.p4_down_channel.1' %time)(p4_in)

            p5_in = layers.Conv2D(num_channels, 1, padding='same', name='backbone_neck.bifpn.%d.p5_down_channel.0.conv' %time)(p5)
            p5_in = layers.BatchNormalization(momentum=0.01, epsilon=1e-3, name='backbone_neck.bifpn.%d.p5_down_channel.1' %time)(p5_in)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in = inputs[0]
            p4_in = inputs[1]
            p5_in = inputs[2]
            p6_in = inputs[3]
            p7_in = inputs[4]

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = K.variable(np.load('weights/backbone_neck.bifpn.%d.p6_w1.npy' %time), name='p6_w1')
        p6_w1 = layers.ReLU()(p6_w1)
        weight = p6_w1 / (K.sum(p6_w1, axis=0) + epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = SeparableConvBlock((Swish(weight[0] * p6_in + weight[1] * layers.UpSampling2D(size=2, interpolation='nearest')(p7_in))),
                                   num_channels, onnx_export=onnx_export, name='backbone_neck.bifpn.%d.conv6_up' %time)

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = K.variable(np.load('weights/backbone_neck.bifpn.%d.p5_w1.npy' %time), name='p5_w1')
        p5_w1 = layers.ReLU()(p5_w1)
        weight = p5_w1 / (K.sum(p5_w1, axis=0) + epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        print(p6_up.shape)
        print(p5_in.shape)
        p5_up = SeparableConvBlock((Swish(weight[0] * p5_in + weight[1] * layers.UpSampling2D(size=2, interpolation='nearest')(p6_up))),
                                   num_channels, onnx_export=onnx_export, name='backbone_neck.bifpn.%d.conv5_up' %time)

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = K.variable(np.load('weights/backbone_neck.bifpn.%d.p4_w1.npy' %time), name='p4_w1')
        p4_w1 = layers.ReLU()(p4_w1)
        weight = p4_w1 / (K.sum(p4_w1, axis=0) + epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = SeparableConvBlock((Swish(weight[0] * p4_in + weight[1] * layers.UpSampling2D(size=2, interpolation='nearest')(p5_up))),
                                   num_channels, onnx_export=onnx_export, name='backbone_neck.bifpn.%d.conv4_up' %time)

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = K.variable(np.load('weights/backbone_neck.bifpn.%d.p3_w1.npy' %time), name='p3_w1')
        p3_w1 = layers.ReLU()(p3_w1)
        weight = p3_w1 / (K.sum(p3_w1, axis=0) + epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = SeparableConvBlock((Swish(weight[0] * p3_in + weight[1] * layers.UpSampling2D(size=2, interpolation='nearest')(p4_up))),
                                   num_channels, onnx_export=onnx_export, name='backbone_neck.bifpn.%d.conv3_up' %time)

        if time==0:
            p4_in = layers.Conv2D(num_channels, 1, padding='same', name='backbone_neck.bifpn.%d.p4_down_channel_2.0.conv' %time)(p4)
            p4_in = layers.BatchNormalization(momentum=0.01, epsilon=1e-3, name='backbone_neck.bifpn.%d.p4_down_channel_2.1' %time)(p4_in)

            p5_in = layers.Conv2D(num_channels, 1, padding='same', name='backbone_neck.bifpn.%d.p5_down_channel_2.0.conv' %time)(p5)
            p5_in = layers.BatchNormalization(momentum=0.01, epsilon=1e-3, name='backbone_neck.bifpn.%d.p5_down_channel_2.1' %time)(p5_in)

        if last:
            print('last')
            return [p3_out]

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = K.variable(np.load('weights/backbone_neck.bifpn.%d.p4_w2.npy' %time), name='p4_w2')
        p4_w2 = layers.ReLU()(p4_w2)
        weight = p4_w2 / (K.sum(p4_w2, axis=0) + epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = SeparableConvBlock((Swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * layers.MaxPooling2D(3, 2, padding='same')(p3_out))),
                                   num_channels, onnx_export=onnx_export, name='backbone_neck.bifpn.%d.conv4_down' %time)

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = K.variable(np.load('weights/backbone_neck.bifpn.%d.p5_w2.npy' %time), name='p5_w2')
        print(np.load('weights/backbone_neck.bifpn.%d.p5_w2.npy' %time))
        p5_w2 = layers.ReLU()(p5_w2)
        weight = p5_w2 / (K.sum(p5_w2, axis=0) + epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = SeparableConvBlock((Swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * layers.MaxPooling2D(3, 2, padding='same')(p4_out))),
                                   num_channels, onnx_export=onnx_export, name='backbone_neck.bifpn.%d.conv5_down' %time)

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = K.variable(np.load('weights/backbone_neck.bifpn.%d.p6_w2.npy' %time), name='p6_w2')
        p6_w2 = layers.ReLU()(p6_w2)
        weight = p6_w2 / (K.sum(p6_w2, axis=0) + epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = SeparableConvBlock((Swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * layers.MaxPooling2D(3, 2, padding='same')(p5_out))),
                                   num_channels, onnx_export=onnx_export, name='backbone_neck.bifpn.%d.conv6_down' %time)

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = K.variable(np.load('weights/backbone_neck.bifpn.%d.p6_w1.npy' %time), name='p7_w2')
        p7_w2 = layers.ReLU()(p7_w2)
        weight = p7_w2 / (K.sum(p7_w2, axis=0) + epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = SeparableConvBlock((Swish(weight[0] * p7_in + weight[1] * layers.MaxPooling2D(3, 2, padding='same')(p6_out))),
                                   num_channels, onnx_export=onnx_export, name='backbone_neck.bifpn.%d.conv7_down' %time)
        
        outs = [p3_out, p4_out, p5_out, p6_out, p7_out]
    else:
        #outs = self._forward(inputs)
        raise Exception('Hey.')

    return outs


def conv3x3(in_planes, out_planes, stride=1, name=None):
    return layers.Conv2D(out_planes, kernel_size=3, strides=stride,
                         padding='same', use_bias=False, name=name)


def BasicBlock(inputs, inplanes, planes, stride=1, downsample=None, name=None):
    expansion = 1

    conv1 = conv3x3(inplanes, planes, stride, name = '%s.conv1' %name)
    bn1 = layers.BatchNormalization(momentum=BN_MOMENTUM, name = '%s.bn1' %name)
    conv2 = conv3x3(planes, planes, name = '%s.conv2' %name)
    bn2 = layers.BatchNormalization(momentum=BN_MOMENTUM, name = '%s.bn2' %name)
    strides = stride

    residual = inputs

    out = conv1(inputs)
    out = bn1(out)
    out = layers.ReLU()(out)

    out = conv2(out)
    out = bn2(out)

    if downsample is not None:
        residual = downsample(inputs)

    out = layers.Add()([out, residual])
    out = layers.ReLU()(out)

    return out


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


def _make_deconv_layers(self, cfg, input_channels):
    dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
    extra = cfg.MODEL.EXTRA
    deconv_cfg = extra.DECONV

    deconv_layers = []
    for i in range(deconv_cfg.NUM_DECONVS):
        if deconv_cfg.CAT_OUTPUT[i]:
            final_output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
                if cfg.LOSS.WITH_AE_LOSS[i] else cfg.MODEL.NUM_JOINTS
            input_channels += final_output_channels
        output_channels = deconv_cfg.NUM_CHANNELS[i]
        deconv_kernel, padding, output_padding = \
            self._get_deconv_cfg(deconv_cfg.KERNEL_SIZE[i])

        _layers = []
        _layers.append(keras.Sequential([
            layers.Conv2DTranspose(
                filters=output_channels,
                kernel_size=deconv_kernel,
                strides=2,
                padding='same',
                use_bias=False),
            layers.BatchNormalization(momentum=BN_MOMENTUM),
            layers.ReLU()
        ]))
        for _ in range(cfg.MODEL.EXTRA.DECONV.NUM_BASIC_BLOCKS):
            _layers.append(keras.Sequential(
                BasicBlock(output_channels, output_channels),
            ))
        deconv_layers.append(keras.Sequential(_layers))
        input_channels = output_channels

    return deconv_layers


def EfficientDetHigherResolutionNet(inputs, cfg, **kwargs):
    #backbone_neck = EfficientDetBackbone(compound_coef=0)

    compound_coef = 0

    backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
    fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
    fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
    pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]

    conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

    '''
    p0 = layers.Conv2D(conv_channel_coef[compound_coef][0], kernel_size=3, padding='same')(inputs)
    p1 = layers.MaxPooling2D(3, 2, padding='same')(p0)
    p2 = layers.MaxPooling2D(3, 2, padding='same')(p1)
    p3 = layers.MaxPooling2D(3, 2, padding='same')(p2)
    p4 = layers.MaxPooling2D(3, 2, padding='same')(p3)
    p4 = layers.Conv2D(conv_channel_coef[compound_coef][1], kernel_size=3, padding='same')(p4)
    p5 = layers.MaxPooling2D(3, 2, padding='same')(p4)
    p5 = layers.Conv2D(conv_channel_coef[compound_coef][2], kernel_size=3, padding='same')(p5)
    '''

    features = EfficientNet(inputs, backbone_compound_coef[compound_coef])

    for _ in range(fpn_cell_repeats[compound_coef]):
        features = BiFPN(features, fpn_num_filters[compound_coef],
                         conv_channel_coef[compound_coef],
                         _,
                         attention=True if compound_coef < 6 else False,
                         use_p8=compound_coef > 7,
                         last=True if _==fpn_cell_repeats[compound_coef]-1 else False)
        
    #fpn_filters = self.backbone_neck.fpn_filters()
    extra = cfg.MODEL.EXTRA
    fpn_filters = fpn_num_filters[compound_coef]
    final_layers = _make_final_layers(cfg, fpn_filters)
    
    input_channels = fpn_filters
    dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
    deconv_cfg = extra.DECONV

    num_deconvs = extra.DECONV.NUM_DECONVS
    concat_layers = [layers.Concatenate() for i in range(num_deconvs)]
    deconv_config = cfg.MODEL.EXTRA.DECONV
    loss_config = cfg.LOSS

    pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    final_outputs = []
    x = features[0]
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
    outputs = EfficientDetHigherResolutionNet(inputs, cfg, **kwargs)
    model = keras.Model(inputs, outputs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model
