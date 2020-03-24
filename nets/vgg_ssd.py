import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr


class VGGSSD:
    def __init__(self, img, num_classes, img_shape):
        self.img = img
        self.num_classes = num_classes
        self.img_shape = img_shape

    def ssd_net(self):
        conv1 = self.conv_block(self.img, 64, 2)
        conv2 = self.conv_block(conv1, 128, 2)
        conv3 = self.conv_block(conv2, 256, 3)
        conv4 = self.conv_block(conv3, 512, 3)

        module11 = self.conv_bn(conv4, 3, 512, 1, 1)
        tmp = self.conv_block(module11, 1024, 5)

        # 10x10
        module13 = fluid.layers.conv2d(tmp, 1024, 1)
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

    def conv_block(self, input, num_filter, groups):
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(input=conv,
                                       num_filters=num_filter,
                                       filter_size=3,
                                       stride=1,
                                       padding=1,
                                       act='relu')
        return fluid.layers.pool2d(input=conv, pool_size=3, pool_type='max', pool_stride=2, pool_padding=1)

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
    ssd_model = VGGSSD(img, num_classes, img_shape)
    return ssd_model.ssd_net()


if __name__ == '__main__':
    data = fluid.data(name='data', shape=[None, 3, 300, 300])
    build_ssd(data, 21, img_shape=[3, 300, 300])
