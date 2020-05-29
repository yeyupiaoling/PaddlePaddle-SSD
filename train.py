import multiprocessing
import os
import shutil
import time
import numpy as np
import paddle.fluid as fluid
import config
from nets import mobilenet_v1_ssd, mobilenet_v2_ssd, vgg_ssd, resnet_ssd
from utils import reader

os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.9'
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'

with open(config.train_list, 'r', encoding='utf-8') as f:
    train_images = len(f.readlines())


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
                    optimizer = fluid.optimizer.Adam(learning_rate=1e-3)
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


def save_model(exe, main_prog, model_path, ssd_out=None, image=None, is_infer_model=False):
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    else:
        os.makedirs(model_path)
    print('save models to %s' % model_path)
    if is_infer_model:
        fluid.io.save_inference_model(dirname=model_path,
                                      feeded_var_names=[image.name],
                                      target_vars=[ssd_out],
                                      executor=exe,
                                      main_program=main_prog)
    else:
        fluid.io.save_persistables(executor=exe,
                                   dirname=model_path,
                                   main_program=main_prog)


def test(epoc_id, best_map, exe, test_prog, map_eval, nmsed_out, image, test_py_reader):
    _, accum_map = map_eval.get_map_var()
    map_eval.reset(exe)
    every_epoc_map = []
    test_py_reader.start()
    try:
        batch_id = 0
        while True:
            test_map, = exe.run(test_prog, fetch_list=[accum_map])
            every_epoc_map.append(test_map[0])
            if batch_id % 100 == 0:
                print("Batch {:d}, map {:.5f}".format(batch_id, test_map[0]))
            batch_id += 1
    except fluid.core.EOFException:
        test_py_reader.reset()
    mean_map = np.mean(every_epoc_map)
    print("Epoc {:d}, test map {:.5f}, Best test map {:.5f}".format(epoc_id, mean_map, best_map))
    if mean_map > best_map:
        best_map = mean_map
        print("Best test map {:.5f}, at present test map {:.5f}".format(best_map, mean_map))
        # save model
        save_model(exe, test_prog, config.infer_model_path, ssd_out=nmsed_out, image=image, is_infer_model=True)
    return best_map, mean_map


def train(data_args, train_file_list, val_file_list):
    if not config.use_gpu:
        devices_num = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        devices_num = fluid.core.get_cuda_device_count()

    batch_size_per_device = config.batch_size // devices_num

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    train_py_reader, loss = build_program(main_prog=train_prog,
                                          startup_prog=startup_prog,
                                          is_train=True)
    test_py_reader, map_eval, nmsed_out, image = build_program(main_prog=test_prog,
                                                               startup_prog=startup_prog,
                                                               is_train=False)

    test_prog = test_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if config.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    # 加载预训练模型或者上一个保存的参数
    if config.persistables_model_path is not None and os.path.exists(config.persistables_model_path):
        def if_exist(var):
            if os.path.exists(os.path.join(config.persistables_model_path, var.name)):
                print('loaded: %s' % var.name)
            return os.path.exists(os.path.join(config.persistables_model_path, var.name))

        print("Loading persistables model: %s" % config.persistables_model_path)
        fluid.io.load_vars(exe, config.persistables_model_path, main_program=train_prog, predicate=if_exist)
    else:
        if config.pretrained_model:
            def if_exist(var):
                if os.path.exists(os.path.join(config.pretrained_model, var.name)):
                    print('loaded: %s' % var.name)
                return os.path.exists(os.path.join(config.pretrained_model, var.name))

            print("Loading pretrained model: %s" % config.pretrained_model)
            fluid.io.load_vars(exe, config.pretrained_model, main_program=train_prog, predicate=if_exist)

    if config.parallel:
        loss.persistable = True
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = False
        train_exe = fluid.ParallelExecutor(main_program=train_prog,
                                           use_cuda=config.use_gpu,
                                           loss_name=loss.name,
                                           build_strategy=build_strategy)

    test_reader = reader.test(data_args, val_file_list, 4)
    test_py_reader.decorate_paddle_reader(test_reader)

    best_map = 0.

    for epoc_id in range(config.epoc_num):
        train_reader = reader.train(data_args,
                                    train_file_list,
                                    batch_size_per_device,
                                    shuffle=True,
                                    use_multiprocess=config.use_multiprocess,
                                    num_workers=config.num_workers)
        train_py_reader.decorate_paddle_reader(train_reader)
        start_time = time.time()
        batch_id = 0
        train_py_reader.start()
        while True:
            try:
                prev_start_time = start_time
                start_time = time.time()
                if config.parallel:
                    loss_v, = train_exe.run(fetch_list=[loss.name])
                else:
                    loss_v, = exe.run(train_prog, fetch_list=[loss])
                loss_v = np.mean(np.array(loss_v))
                if batch_id % 100 == 0:
                    print("Epoc: {:d}, batch: {:d}, loss: {:.5f}, batch/second: {:.5f}".format(
                        epoc_id, batch_id, loss_v, start_time - prev_start_time))
                batch_id += 1
            except (fluid.core.EOFException, StopIteration):
                train_reader().close()
                train_py_reader.reset()
                break

        # run test
        best_map, mean_map = test(epoc_id, best_map, exe, test_prog, map_eval, nmsed_out, image, test_py_reader)
        # save model
        save_model(exe, train_prog, config.persistables_model_path, is_infer_model=False)


if __name__ == '__main__':
    config.print_value()
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
