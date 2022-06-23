#!/usr/bin/python
# -*- coding:utf-8 -*-

import os, sys, time
import numpy as np
import onnx
import onnxsim
import onnx_graphsurgeon as gs
import tensorrt as trt
from utils.common import allocate_buffers, do_inference_v2
from utils.efficientdet_build_engine import EngineCalibrator
from utils.image_batcher import ImageBatcher
from panoptic_bev.utils import plogging
plogging.init("./", "onnx2trt_test")
logger = plogging.get_logger()

class Onnx2TRT(object):
    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        logger.info("TensorRT version: {}".format(trt.__version__))
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = workspace * (2 ** 30)

        self.network = None
        self.parser = None
        self.context = None
        self.engine = None

    def __del__(self):
        print("Onnx2TRT __del__")

    def destruct(self):
        if self.context:
            del self.context
        if self.engine:
            del self.engine

    def print_version(self):
        # print("torch: {}".format(torch.__version__))
        logger.info("onnx version: {}".format(onnx.__version__))
        logger.info("trt version: {}".format(trt.__version__))

    def simplify_onnx_model(self, model_path):
        model = onnx.load(model_path)
        logger.debug("simplify onnx model: {}".format(model_path))
        # logger.info('onnx model graph is:\n{}'.format(model.graph))
        model_sim, check = onnxsim.simplify(model)
        logger.debug("onnxsim check: {}".format(check))
        new_path = model_path + ".sim"
        onnx.save(model_sim, new_path)
        logger.debug("simplify model saved to: {}".format(new_path))

    def create_network(self, onnx_path, batch_size=1, dynamic_batch_size=None):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        :param batch_size: Static batch size to build the engine with.
        :param dynamic_batch_size: Dynamic batch size to build the engine with, if given,
        batch_size is ignored, pass as a comma-separated string or int list as MIN,OPT,MAX
        """
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                logger.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    logger.error(self.parser.get_error(error))
                sys.exit(1)
        logger.info("Network Description: ")
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        for input in inputs:
            self.batch_size = input.shape[0]
            logger.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            logger.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size
        logger.info("builder support fp16: {}, int8: {}".format(self.builder.platform_has_fast_fp16, self.builder.platform_has_fast_int8))

    def make_calibrator(self, calib_input=None, calib_cache="./calib.cache", calib_num_images=5000, calib_batch_size=8):
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        self.config.int8_calibrator = EngineCalibrator(calib_cache)
        if calib_cache is None or not os.path.exists(calib_cache):
            calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
            calib_dtype = trt.nptype(inputs[0].dtype)
            self.config.int8_calibrator.set_image_batcher(
                ImageBatcher(calib_input, calib_shape, calib_dtype, max_num_images=calib_num_images,
                            exact_batches=True, shuffle_files=True))

    def create_engine(self, engine_path, precision="fp16"):
        self.config = self.builder.create_builder_config()
        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                logger.warning("FP16 is not supported natively on this platform/device")
                return None
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                logger.warning("INT8 is not supported natively on this platform/device")
                return None
            else:
                if self.builder.platform_has_fast_fp16:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    self.config.set_flag(trt.BuilderFlag.FP16)
                else:
                    self.config.set_flag(trt.BuilderFlag.INT8)
                    # using nuscenes-mini images for calibration
                    self.make_calibrator(calib_input="/home/hugoliu/github/dataset/nuscenes/mini/samples/CAM_FRONT")

        self.config.max_workspace_size = 1<<30 #1GB
        # profile = self.builder.create_optimization_profile()
        # profile.set_shape('input', (1, 1, 4, 4), (2, 1, 4, 4), (4, 1, 4, 4))
        # profile.set_shape('grid', (1, 4, 4, 2), (2, 4, 4, 2), (4, 4, 4, 2))
        # self.config.add_optimization_profile(profile)
        logger.info("build trt engine with config {}".format(self.config))
        self.engine = self.builder.build_engine(self.network, self.config)
        return self.engine

    def save_engine(self, engine_path):
        assert self.engine
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(engine_path, "wb") as f:
            logger.info("Serializing engine to file: {:}".format(engine_path))
            f.write(self.engine.serialize())
    
    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        logger.info("engine loaded, name: {}".format(self.engine.name))
        # inspector only avaliable since TRT-8.4
        inspector = self.engine.create_engine_inspector()
        inspector.execution_context = self.context
        logger.info(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        return self.engine

    def run_engine(self, inputs_np):
        assert self.engine
        self.context = self.engine.create_execution_context()
        inputs, outputs, bindings, stream = allocate_buffers(self.engine)
        for idx in range(len(inputs)):
            inputs[idx].host = inputs_np[idx]
        trt_outputs = do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        return trt_outputs

    def benchmark(self, trt_path, inputs_np, nwarmup=10, nruns=1000):
        self.load_engine(trt_path)
        assert self.engine
        self.context = self.engine.create_execution_context()
        inputs, outputs, bindings, stream = allocate_buffers(self.engine)
        logger.debug("Warm up ...")
        for _ in range(nwarmup):
            for idx in range(len(inputs)):
                inputs[idx].host = inputs_np[idx]
            trt_outputs = do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        logger.debug("Start timing ...")
        timings = []
        for i in range(1, nruns+1):
            start_time = time.time()
            for idx in range(len(inputs)):
                inputs[idx].host = inputs_np[idx]
            trt_outputs = do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                logger.debug('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))
        logger.debug('Average batch time: %.2f ms'%(np.mean(timings)*1000))

    '''
        Name: Unnamed Network 0 | Explicit Batch Engine (498 layers)
        ---- 1 Engine Input(s) ----
        {x.1 [dtype=float32, shape=(1, 3, 448, 768)]}
        ---- 5 Engine Output(s) ----
        {17831 [dtype=float32, shape=(1, 160, 112, 192)],
        18048 [dtype=float32, shape=(1, 160, 56, 96)],
        18265 [dtype=float32, shape=(1, 160, 28, 48)],
        18482 [dtype=float32, shape=(1, 160, 14, 24)],
        18695 [dtype=float32, shape=(1, 160, 7, 12)]}
    '''
    def test_encoder(self, model_path, trt_path, only_build_engine=False):
        model = onnx.load(model_path)
        try:
            onnx.checker.check_model(model)
            # logger.info(onnx.helper.printable_graph(model.graph))
            # logger.info('onnx model graph is:\n{}'.format(model.graph))
        except Exception as e:
            logger.error("onnx check model error: {}".format(e))
            return False
        if only_build_engine:
            self.create_network(onnx_path=model_path)
            self.create_engine(engine_path=trt_path, precision="fp16")
            self.save_engine(engine_path=trt_path)
            self.destruct()
            return True
        else:
            image = np.random.rand(1, 3, 448, 768).astype(np.float32)
            self.load_engine(trt_path)
            results = self.run_engine([image])
            # for feat in results:
            #     logger.info("{} output: {}".format(trt_path, feat.shape))
            # self.benchmark(trt_path, [image])
            # delete context & engine to avoid segment fault in the end
            self.destruct()
            return True

    '''
        Name: Unnamed Network 0 | Explicit Batch Engine (100 layers)
        ---- 4 Engine Input(s) ----
        {input.117 [dtype=float32, shape=(1, 256, 224, 384)],
        input.101 [dtype=float32, shape=(1, 256, 112, 192)],
        input.43 [dtype=float32, shape=(1, 256, 56, 96)],
        input.1 [dtype=float32, shape=(1, 256, 28, 48)]}
        ---- 2 Engine Output(s) ----
        {357 [dtype=float32, shape=(1, 10, 896, 1536)],
        363 [dtype=int32, shape=(896, 1536)]}
    '''
    def test_sem_head(self, model_path, trt_path, only_build_engine=False):
        model = onnx.load(model_path)
        try:
            onnx.checker.check_model(model)
            # logger.info(onnx.helper.printable_graph(model.graph))
            # logger.info('onnx model graph is:\n{}'.format(model.graph))
        except Exception as e:
            logger.error("onnx check model error: {}".format(e))
            return False
        if only_build_engine:
            self.create_network(onnx_path=model_path)
            self.create_engine(engine_path=trt_path, precision="fp16")
            self.save_engine(engine_path=trt_path)
            self.destruct()
            return True
        else:
            ms_bev_0 = np.random.rand(1, 256, 224, 384).astype(np.float32)
            ms_bev_1 = np.random.rand(1, 256, 112, 192).astype(np.float32)
            ms_bev_2 = np.random.rand(1, 256, 56, 96).astype(np.float32)
            ms_bev_3 = np.random.rand(1, 256, 28, 48).astype(np.float32)
            self.load_engine(trt_path)
            results = self.run_engine([ms_bev_0, ms_bev_1, ms_bev_2, ms_bev_3])
            for feat in results:
                logger.info("{} output: {}".format(trt_path, feat.shape))
            # self.benchmark(trt_path, [ms_bev_0, ms_bev_1, ms_bev_2, ms_bev_3])
            # delete context & engine to avoid segment fault in the end
            self.destruct()
            return True

    def test_rpn_neck(self, model_path, trt_path, only_build_engine=True):
        model = onnx.load(model_path)
        try:
            onnx.checker.check_model(model)
            logger.info(onnx.helper.printable_graph(model.graph))
            # logger.info('onnx model graph is:\n{}'.format(model.graph))
        except Exception as e:
            logger.error("onnx check model error: {}".format(e))
            return False
        if only_build_engine:
            self.create_network(onnx_path=model_path)
            self.create_engine(engine_path=trt_path, precision="fp16")
            self.save_engine(engine_path=trt_path)
            self.destruct()
            return True
        else:
            self.load_engine(trt_path)
            # results = self.run_engine([ms_bev_0, ms_bev_1, ms_bev_2, ms_bev_3])
            # for feat in results:
            #     logger.info("{} output: {}".format(trt_path, feat.shape))
            # self.benchmark(trt_path, [ms_bev_0, ms_bev_1, ms_bev_2, ms_bev_3])
            # delete context & engine to avoid segment fault in the end
            self.destruct()
            return True

    def test_po_fusion(self, model_path, trt_path, only_build_engine=True):
        model = onnx.load(model_path)
        try:
            onnx.checker.check_model(model)
            logger.info(onnx.helper.printable_graph(model.graph))
            # logger.info('onnx model graph is:\n{}'.format(model.graph))
        except Exception as e:
            logger.error("onnx check model error: {}".format(e))
            return False
        if only_build_engine:
            self.create_network(onnx_path=model_path)
            self.create_engine(engine_path=trt_path, precision="fp16")
            self.save_engine(engine_path=trt_path)
            self.destruct()
            return True
        else:
            self.load_engine(trt_path)
            # results = self.run_engine([ms_bev_0, ms_bev_1, ms_bev_2, ms_bev_3])
            # for feat in results:
            #     logger.info("{} output: {}".format(trt_path, feat.shape))
            # self.benchmark(trt_path, [ms_bev_0, ms_bev_1, ms_bev_2, ms_bev_3])
            # delete context & engine to avoid segment fault in the end
            self.destruct()
            return True

    def test_demo(self, model_path, trt_path, only_build_engine=True):
        model = onnx.load(model_path)
        try:
            onnx.checker.check_model(model)
            logger.info(onnx.helper.printable_graph(model.graph))
            logger.info('onnx model graph is:\n{}'.format(model.graph))
        except Exception as e:
            logger.error("onnx check model error: {}".format(e))
            return False
        if only_build_engine:
            self.create_network(onnx_path=model_path)
            self.create_engine(engine_path=trt_path, precision="fp16")
            self.save_engine(engine_path=trt_path)
            self.destruct()
            return True
        else:
            self.load_engine(trt_path)
            logits = np.random.rand(10, 4, 28, 28).astype(np.float32)
            indices = np.random.rand(10).astype(np.int64)
            results = self.run_engine([logits, indices])
            logger.info("{} results: {}".format(trt_path, len(results)))
            for feat in results:
                tmp = feat.reshape(-1, 4, 28, 28)
                logger.info("{} output: {}, reshaped: {}".format(trt_path, feat.shape, tmp.shape))
            # self.benchmark(trt_path, [ms_bev_0, ms_bev_1, ms_bev_2, ms_bev_3])
            # delete context & engine to avoid segment fault in the end
            self.destruct()
            return True

    def test_transformer(self, model_path, trt_path, only_build_engine=True):
        model = onnx.load(model_path)
        try:
            onnx.checker.check_model(model)
            logger.info(onnx.helper.printable_graph(model.graph))
            # logger.info('onnx model graph is:\n{}'.format(model.graph))
        except Exception as e:
            logger.error("onnx check model error: {}".format(e))
            return False
        if only_build_engine:
            self.create_network(onnx_path=model_path)
            self.create_engine(engine_path=trt_path, precision="fp16")
            self.save_engine(engine_path=trt_path)
            self.destruct()
            return True
        else:
            self.load_engine(trt_path)
            # results = self.run_engine([ms_bev_0, ms_bev_1, ms_bev_2, ms_bev_3])
            # for feat in results:
            #     logger.info("{} output: {}".format(trt_path, feat.shape))
            # self.benchmark(trt_path, [ms_bev_0, ms_bev_1, ms_bev_2, ms_bev_3])
            # delete context & engine to avoid segment fault in the end
            self.destruct()
            return True

    def test_roi_head(self, model_path, trt_path, only_build_engine=True):
        model = onnx.load(model_path)
        try:
            onnx.checker.check_model(model)
            logger.info(onnx.helper.printable_graph(model.graph))
            # logger.info('onnx model graph is:\n{}'.format(model.graph))
        except Exception as e:
            logger.error("onnx check model error: {}".format(e))
            return False
        if only_build_engine:
            self.create_network(onnx_path=model_path)
            self.create_engine(engine_path=trt_path, precision="fp16")
            self.save_engine(engine_path=trt_path)
            self.destruct()
            return True
        else:
            self.load_engine(trt_path)
            # results = self.run_engine([ms_bev_0, ms_bev_1, ms_bev_2, ms_bev_3])
            # for feat in results:
            #     logger.info("{} output: {}".format(trt_path, feat.shape))
            # self.benchmark(trt_path, [ms_bev_0, ms_bev_1, ms_bev_2, ms_bev_3])
            # delete context & engine to avoid segment fault in the end
            self.destruct()
            return True

if __name__ == "__main__":
    resnet50_path = "../resnet50-v1-12/resnet50-v1-12.onnx"
    #run "polygraphy surgeon sanitize model.onnx --fold-constants --output model_folded.onnx" first
    encoder_onnx_path = "../../onnx/body_encoder_folded.onnx"
    encoder_trt_path = "../body_encoder_int8.trt"

    transformer_onnx_path = "../../onnx/transformer_op13.onnx"
    transformer_trt_path = "../../onnx/transformer_fp16.trt"

    roi_onnx_path = "../../onnx/roi_algo_op13.onnx"
    roi_trt_path = "../../onnx/roi_algo_fp16.trt"

    sem_head_onnx_path = "../../onnx/sem_algo_fold.onnx"
    sem_head_trt_path = "../sem_algo_fp16.trt"

    rpn_neck_onnx_path = "../../onnx/rpn_algo_op13.onnx"
    # rpn_neck_onnx_path = "../../onnx/rpn_algo_fold.onnx"
    rpn_neck_trt_path = "../rpn_algo_fp16.trt"

    po_fusion_onnx_path = "../../onnx/po_fusion_op13.onnx"
    po_fusion_trt_path = "../po_fusion_fp16.trt"

    demo_onnx_path = "./demo/trt_demo.onnx"
    demo_trt_path = "./demo/trt_demo_fp16.trt"

    onnxtrt = Onnx2TRT(verbose=True)
    # onnxtrt.test_encoder(encoder_onnx_path, encoder_trt_path)
    # onnxtrt.test_sem_head(sem_head_onnx_path, sem_head_trt_path, only_build_engine=False)
    # onnxtrt.test_rpn_neck(rpn_neck_onnx_path, rpn_neck_trt_path)
    # onnxtrt.test_po_fusion(po_fusion_onnx_path, po_fusion_trt_path)
    onnxtrt.test_demo(demo_onnx_path, demo_trt_path, only_build_engine=False)
    # onnxtrt.test_transformer(transformer_onnx_path, transformer_trt_path)
    # onnxtrt.test_roi_head(roi_onnx_path, roi_trt_path)
