import numpy as np
import onnx
import denseFCN2
import mqbench
import torchvision.models as models                           # for example model
from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.convert_deploy import convert_deploy

if __name__ == '__main__':
    model = denseFCN2.normal_denseFCN(bn_in='bn')  # 模型1
    model.eval()
    backend = BackendType.OPENVINO
    model = prepare_by_platform(model, backend)
    enable_calibration(model)
    enable_quantization(model)
    input_shape = {'data': [10, 3, 224, 224]}
    convert_deploy(model, backend, input_shape)

