import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr


class MobileNetV2SSD:
    def __init__(self, img, num_classes, img_shape):
        self.img = img
        self.num_classes = num_classes
        self.img_shape = img_shape

    def ssd_net(self, scale=1.0):
        # 300x300
        bottleneck_params_list = [(1, 16, 1, 1),
                                  (6, 24, 2, 2),
                                  (6, 32, 3, 2),
                                  (6, 64, 4, 2),
                                  (6, 96, 3, 1)]

        # conv1
        input = self.conv_bn_layer(input=self.img,
                                   num_filters=int(32 * scale),
                                   filter_size=3,
                                   stride=2,
                                   padding=1,
                                   if_act=True)

        # bottleneck sequences
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            input = self.invresi_blocks(input=input, in_c=in_c, t=t, c=int(c * scale), n=n, s=s)
            in_c = int(c * scale)
        # 19x19
        module11 = input
        tmp = self.invresi_blocks(input=input, in_c=in_c, t=6, c=int(512 * scale), n=3, s=2)

        # 10x10
        module13 = self.invresi_blocks(input=tmp, in_c=int(512 * scale), t=6, c=int(1024 * scale), n=1, s=1)
        module14 = self.extra_block(module13, 256, 512, 1)
        # 5x5
        module15 = self.extra_block(module14, 128, 256, 1)
        # 3x3
        module16 = self.extra_block(module15, 128, 256, 1)
        # 2x2
        module17 = self.extra_block(module16, 64, 128, 1)

        mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
            inputs=[module11, module13, module14, module15, module16, module17],
            image=self.img,
            num_classes=self.num_classes,
            min_ratio=20,
            max_ratio=90,
            min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
            max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
            aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.], [2., 3.]],
            base_size=self.img_shape[2],
            offset=0.5,
            flip=True)

        return mbox_locs, mbox_confs, box, box_var

    def conv_bn_layer(self, input, filter_size, num_filters, stride, padding, num_groups=1, if_act=True,
                      use_cudnn=True):
        parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=num_groups,
                                   use_cudnn=use_cudnn,
                                   param_attr=parameter_attr,
                                   bias_attr=False)
        bn = fluid.layers.batch_norm(input=conv)
        if if_act:
            return fluid.layers.relu6(bn)
        else:
            return bn

    def shortcut(self, input, data_residual):
        return fluid.layers.elementwise_add(input, data_residual)

    def inverted_residual_unit(self,
                               input,
                               num_in_filter,
                               num_filters,
                               ifshortcut,
                               stride,
                               filter_size,
                               padding,
                               expansion_factor):
        num_expfilter = int(round(num_in_filter * expansion_factor))

        channel_expand = self.conv_bn_layer(input=input,
                                            num_filters=num_expfilter,
                                            filter_size=1,
                                            stride=1,
                                            padding=0,
                                            num_groups=1,
                                            if_act=True)

        bottleneck_conv = self.conv_bn_layer(input=channel_expand,
                                             num_filters=num_expfilter,
                                             filter_size=filter_size,
                                             stride=stride,
                                             padding=padding,
                                             num_groups=num_expfilter,
                                             if_act=True,
                                             use_cudnn=False)

        linear_out = self.conv_bn_layer(input=bottleneck_conv,
                                        num_filters=num_filters,
                                        filter_size=1,
                                        stride=1,
                                        padding=0,
                                        num_groups=1,
                                        if_act=False)
        if ifshortcut:
            out = self.shortcut(input=input, data_residual=linear_out)
            return out
        else:
            return linear_out

    def invresi_blocks(self, input, in_c, t, c, n, s):
        first_block = self.inverted_residual_unit(input=input,
                                                  num_in_filter=in_c,
                                                  num_filters=c,
                                                  ifshortcut=False,
                                                  stride=s,
                                                  filter_size=3,
                                                  padding=1,
                                                  expansion_factor=t)

        last_residual_block = first_block
        last_c = c

        for i in range(1, n):
            last_residual_block = self.inverted_residual_unit(input=last_residual_block,
                                                              num_in_filter=last_c,
                                                              num_filters=c,
                                                              ifshortcut=True,
                                                              stride=1,
                                                              filter_size=3,
                                                              padding=1,
                                                              expansion_factor=t)
        return last_residual_block

    def conv_bn(self, input, filter_size, num_filters, stride, padding, num_groups=1, act='relu', use_cudnn=True):
        parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=num_groups,
                                   use_cudnn=use_cudnn,
                                   param_attr=parameter_attr,
                                   bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def extra_block(self, input, num_filters1, num_filters2, num_groups):
        # 1x1 conv
        pointwise_conv = self.conv_bn(input=input,
                                      filter_size=1,
                                      num_filters=int(num_filters1),
                                      stride=1,
                                      num_groups=int(num_groups),
                                      padding=0)

        # 3x3 conv
        normal_conv = self.conv_bn(input=pointwise_conv,
                                   filter_size=3,
                                   num_filters=int(num_filters2),
                                   stride=2,
                                   num_groups=int(num_groups),
                                   padding=1)
        return normal_conv


def build_ssd(img, num_classes, img_shape):
    ssd_model = MobileNetV2SSD(img, num_classes, img_shape)
    return ssd_model.ssd_net()


if __name__ == '__main__':
    data = fluid.data(name='data', shape=[None, 3, 300, 300])
    build_ssd(data, 21, img_shape=[3, 300, 300])
