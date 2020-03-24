import multiprocessing
import os
import shutil
import time
from nets import mobilenet_v1_ssd, mobilenet_v2_ssd, vgg_ssd, resnet_ssd
import numpy as np
import paddle.fluid as fluid
import config
from utils import reader

with open(config.train_list, 'r', encoding='utf-8') as f:
    train_images = len(f.readlines())


def optimizer_setting():
    lr = 0.0001
    iters = train_images // config.batch_size
    boundaries = [i * iters for i in config.lr_epochs]
    values = [i * lr for i in config.lr_decay]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005), )

    return optimizer


def build_program(main_prog, startup_prog, is_train):
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
            if is_train:
                with fluid.unique_name.guard("train"):
                    loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box, box_var)
                    loss = fluid.layers.reduce_sum(loss)
                    optimizer = optimizer_setting()
                    optimizer.minimize(loss)
                outs = [py_reader, loss]
            else:
                with fluid.unique_name.guard("inference"):
                    nmsed_out = fluid.layers.detection_output(locs, confs, box, box_var, nms_threshold=0.45)
                    map_eval = fluid.metrics.DetectionMAP(
                        nmsed_out,
                        gt_label,
                        gt_box,
                        difficult,
                        config.class_num,
                        overlap_threshold=0.5,
                        evaluate_difficult=False,
                        ap_version=config.ap_version)
                # nmsed_out and image is used to save mode for inference
                outs = [py_reader, map_eval, nmsed_out, image]
    return outs


def test(exe, test_prog, map_eval, test_py_reader):
    _, accum_map = map_eval.get_map_var()
    map_eval.reset(exe)
    every_epoc_map = []
    test_py_reader.start()
    try:
        batch = 0
        while True:
            test_map, = exe.run(test_prog, fetch_list=[accum_map])
            every_epoc_map.append(test_map)
            if batch % 10 == 0:
                print("Batch {0}, map {1}".format(batch, test_map))
            batch += 1
    except fluid.core.EOFException:
        test_py_reader.reset()
    finally:
        test_py_reader.reset()
    mean_map = np.mean(every_epoc_map)
    return mean_map


def save_infer_model(exe, main_prog, ssd_out, model_save_dir):
    if not os.path.basename(model_save_dir):
        os.makedirs(os.path.basename(model_save_dir))
    if os.path.exists(model_save_dir):
        shutil.rmtree(model_save_dir)
    print('save models to %s' % model_save_dir)
    fluid.io.save_inference_model(dirname=model_save_dir,
                                  feeded_var_names=[],
                                  target_vars=[ssd_out],
                                  executor=exe,
                                  main_program=main_prog,
                                  model_filename='model.paddle',
                                  params_filename='params.paddle')


def train(data_args, train_file_list, val_file_list):
    if config.use_gpu:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    batch_size_per_device = config.batch_size // devices_num

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    train_py_reader, loss = build_program(main_prog=train_prog,
                                          startup_prog=startup_prog,
                                          is_train=True)
    test_py_reader, map_eval, nmsed_out, _ = build_program(main_prog=test_prog,
                                                           startup_prog=startup_prog,
                                                           is_train=False)

    test_prog = test_prog.clone(for_test=True)

    transpiler = fluid.contrib.QuantizeTranspiler(weight_bits=8,
                                                  activation_bits=8,
                                                  activation_quantize_type='abs_max',
                                                  weight_quantize_type='abs_max')

    transpiler.training_transpile(train_prog, startup_prog)
    transpiler.training_transpile(test_prog, startup_prog)

    place = fluid.CUDAPlace(0) if config.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if config.persistables_model_path:
        def if_exist(var):
            if os.path.exists(os.path.join(config.pretrained_model, var.name)):
                print('loaded: %s' % var.name)
            return os.path.exists(os.path.join(config.persistables_model_path, var.name))

        fluid.io.load_vars(exe, config.persistables_model_path, main_program=train_prog, predicate=if_exist)
    else:
        print('There is no init model.')

    # transpiler.freeze_program(test_prog, place)

    if config.parallel:
        train_exe = fluid.ParallelExecutor(main_program=train_prog,
                                           use_cuda=True if config.use_gpu else False, loss_name=loss.name)

    train_reader = reader.train(data_args,
                                train_file_list,
                                batch_size_per_device,
                                shuffle=True,
                                num_workers=config.num_workers)
    test_reader = reader.test(data_args, val_file_list, config.batch_size)
    train_py_reader.decorate_paddle_reader(train_reader)
    test_py_reader.decorate_paddle_reader(test_reader)

    train_py_reader.start()
    for epoc_id in range(20):
        if epoc_id == 0:
            # test quantized model without quantization-aware training.
            test_map = test(exe, test_prog, map_eval, test_py_reader)
            print("without quantization-aware training test map {0}".format(test_map))
        batch_id = 0
        train_py_reader.start()
        while True:
            try:
                # train
                start_time = time.time()
                if config.parallel:
                    outs = train_exe.run(fetch_list=[loss.name])
                else:
                    outs = exe.run(train_prog, fetch_list=[loss])
                end_time = time.time()
                avg_loss = np.mean(np.array(outs[0]))
                if batch_id % 10 == 0:
                    print("Epoc: {:d}, batch: {:d}, loss: {:.6f}, batch/second: {:.5f}".format(
                        epoc_id, batch_id, avg_loss, end_time - start_time))
                batch_id += 1
            except (fluid.core.EOFException, StopIteration):
                train_reader().close()
                train_py_reader.reset()
                break
        # run test
        test_map = test(exe, test_prog, map_eval, test_py_reader)
        print("Epoc {0}, test map {1}".format(epoc_id, test_map))
        # save infer model
        save_infer_model(exe, test_prog, nmsed_out, config.quant_infer_model_path)


if __name__ == '__main__':
    data_args = reader.Settings(label_file_path=config.label_file,
                                resize_h=config.image_shape[1],
                                resize_w=config.image_shape[2],
                                mean_value=config.img_mean,
                                std_value=config.img_std,
                                apply_distort=True,
                                apply_expand=True,
                                ap_version=config.ap_version)
    train(data_args=data_args,
          train_file_list=config.train_list,
          val_file_list=config.test_list)
