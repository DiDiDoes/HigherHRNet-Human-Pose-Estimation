# Author: Zylo117

from tensorflow import keras
from tensorflow.keras import layers

from .efficientdet.model import BiFPN, EfficientNet


class EfficientDetBackbone(keras.Model):
    def __init__(self, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]

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

        #self.input_to_p0 = layers.Conv2D(self.fpn_num_filters[self.compound_coef], kernel_size=3, padding='same')
        #self.p0_to_p1 = layers.MaxPooling2D(3, 2, padding='same')
        #self.p1_to_p2 = layers.MaxPooling2D(3, 2, padding='same')
        #self.p2_to_p3 = layers.MaxPooling2D(3, 2, padding='same')
        #self.p3_to_p4 = layers.MaxPooling2D(3, 2, padding='same')
        #self.p4_to_p5 = layers.MaxPooling2D(3, 2, padding='same')

        self.bifpn = [BiFPN(self.fpn_num_filters[self.compound_coef],
                            conv_channel_coef[compound_coef],
                            True if _ == 0 else False,
                            attention=True if compound_coef < 6 else False,
                            use_p8=compound_coef > 7)
                      for _ in range(self.fpn_cell_repeats[compound_coef])]

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def fpn_filters(self):
        return self.fpn_num_filters[self.compound_coef]

    '''
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    '''

    def call(self, inputs):
        max_size = inputs.shape[-1]
        _, p3, p4, p5 = self.backbone_net(inputs)
        #p0 = self.input_to_p0(inputs)
        #p1 = self.p0_to_p1(p0)
        #p2 = self.p1_to_p2(p1)
        #p3 = self.p2_to_p3(p2)
        #p4 = self.p3_to_p4(p3)
        #p5 = self.p4_to_p5(p4)

        features = (p3, p4, p5)
        for fpn in self.bifpn:
            features = fpn(features)

        return features

    '''
    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')
    '''
