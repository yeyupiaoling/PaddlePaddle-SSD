import os
import paddle.fluid as fluid
import config
import numpy as np
from nets import mobilenet_v1_ssd, mobilenet_v2_ssd, vgg_ssd, resnet_ssd
from utils import reader


def build_program(main_prog, startup_prog):
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(capacity=64,
                                           shapes=[[-1] + config.image_shape, [-1, 4], [-1, 1], [-1, 1]],
                                           lod_levels=[0, 1, 1, 1],
                                           dtypes=["float32", "float32", "int32", "int32"],
                                           use_double_buffer=True)
        with fluid.unique_name.guard():
            image, gt_box, gt_label, difficult = fluid.layers.read_file(py_reader)
            if config.use_model == 'mobilenet_v2_ssd':
                locs, confs, box, box_var = mobilenet_v2_ssd.build_ssd(image, config.class_num, config.image_shape)
            elif config.use_model == 'mobilenet_v1_ssd':
                locs, confs, box, box_var = mobilenet_v1_ssd.build_ssd(image, config.class_num, config.image_shape)
            elif config.use_model == 'vgg_ssd':
                locs, confs, box, box_var = vgg_ssd.build_ssd(image, config.class_num, config.image_shape)
            elif config.use_model == 'resnet_ssd':
                locs, confs, box, box_var = resnet_ssd.build_ssd(image, config.class_num, config.image_shape)
            else:
                raise Exception('not have %s model' % config.use_model)
            nmsed_out = fluid.layers.detection_output(locs, confs, box, box_var, nms_threshold=config.nms_threshold)
            with fluid.program_guard(main_prog):
                map = fluid.metrics.DetectionMAP(
                    nmsed_out,
                    gt_label,
                    gt_box,
                    difficult,
                    config.class_num,
                    overlap_threshold=0.5,
                    evaluate_difficult=False,
                    ap_version=config.ap_version)
    return py_reader, map


def eval(data_args, test_list, batch_size, model_dir=None):
    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    test_py_reader, map_eval = build_program(main_prog=test_prog,
                                             startup_prog=startup_prog)
    test_prog = test_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if config.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    def if_exist(var):
        if os.path.exists(os.path.join(config.pretrained_model, var.name)):
            print('loaded: %s' % var.name)
        return os.path.exists(os.path.join(model_dir, var.name))

    fluid.io.load_vars(exe, model_dir, main_program=test_prog, predicate=if_exist)

    test_reader = reader.test(data_args, test_list, batch_size=batch_size)
    test_py_reader.decorate_paddle_reader(test_reader)

    _, accum_map = map_eval.get_map_var()
    map_eval.reset(exe)
    every_epoc_map = []
    test_py_reader.start()
    try:
        batch_id = 0
        while True:
            test_map, = exe.run(test_prog, fetch_list=[accum_map])
            if batch_id % 10 == 0:
                every_epoc_map.append(test_map[0])
                print("Batch {:d}, map {:.5f}".format(batch_id, test_map[0]))
            batch_id += 1
    except (fluid.core.EOFException, StopIteration):
        test_py_reader.reset()
    mean_map = np.mean(every_epoc_map)
    print("Test model {:s}, map {:.5f}".format(model_dir, mean_map))


if __name__ == '__main__':
    if not os.path.exists(config.persistables_model_path):
        raise ValueError("The model path [%s] does not exist." % (config.persistables_model_path))

    data_args = reader.Settings(label_file_path=config.label_file,
                                resize_h=config.image_shape[1],
                                resize_w=config.image_shape[2],
                                mean_value=config.img_mean,
                                std_value=config.img_std,
                                apply_distort=False,
                                apply_expand=False,
                                ap_version=config.ap_version)
    eval(data_args=data_args,
         test_list=config.test_list,
         batch_size=config.batch_size,
         model_dir=config.persistables_model_path)
