import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr


class ResNetSSD:
    def __init__(self, img, num_classes, img_shape):
        self.img = img
        self.num_classes = num_classes
        self.img_shape = img_shape

    def ssd_net(self):

        depth = [3, 4, 6, 3]
        num_filters = [64, 128, 256, 512]
        conv = self.conv_bn_layer(input=self.img,
                                  num_filters=64,
                                  filter_size=7,
                                  stride=2,
                                  act='relu')
        conv = fluid.layers.pool2d(input=conv,
                                   pool_size=3,
                                   pool_stride=1,
                                   pool_padding=1,
                                   pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                stride = 2 if i == 0 and block != 0 else 1
                conv = self.bottleneck_block(input=conv,
                                             num_filters=num_filters[block],
                                             stride=stride)
        module11 = conv

        # 10x10
        module13 = self.conv_bn_layer(conv, 1024, 3, 2, act='relu')
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

    def conv_bn_layer(self, input, num_filters, filter_size, stride=1, groups=1, act=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups)

        return fluid.layers.batch_norm(input=conv, act=act)

    def shortcut(self, input, ch_out, stride, is_first):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1 or is_first == True:
            return self.conv_bn_layer(input, ch_out, 1, stride)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride):
        conv0 = self.conv_bn_layer(input=input,
                                   num_filters=num_filters,
                                   filter_size=1,
                                   act='relu')
        conv1 = self.conv_bn_layer(input=conv0,
                                   num_filters=num_filters,
                                   filter_size=3,
                                   stride=stride,
                                   act='relu')
        conv2 = self.conv_bn_layer(input=conv1,
                                   num_filters=num_filters * 4,
                                   filter_size=1,
                                   act=None)

        short = self.shortcut(input,
                              num_filters * 4,
                              stride,
                              is_first=False)

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')

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
    ssd_model = ResNetSSD(img, num_classes, img_shape)
    return ssd_model.ssd_net()


if __name__ == '__main__':
    data = fluid.data(name='data', shape=[None, 3, 300, 300])
    build_ssd(data, 21, img_shape=[3, 300, 300])
