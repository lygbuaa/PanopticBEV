from pickle import NONE
import torch, sys
import torch.nn.functional as F
sys.path.append("/home/hugoliu/github/PanopticBEV/onnx/script")
from onnx_wrapper import OnnxWrapper


class TrtDemo(torch.nn.Module):
    def __init__(self, digits_list):
        super(TrtDemo, self).__init__()
        self.digits_list = digits_list

    def forward(self, idx:int):
        result = self.digits_list[idx]
        return result


def test_onnx(onnx_model_path):
    model_onnx = OnnxWrapper()
    model_onnx.load_onnx_model(onnx_model_path)
    for idx in range(10):
        result = model_onnx.run([idx])
        print("onnx result: {}".format(result))

if __name__ == "__main__":
    print("torch version: {}".format(torch.__version__))
    logits = torch.rand(size=[10, 3], dtype=torch.float)

    model = TrtDemo(logits)
    model.eval()
    # model_jit = torch.jit.script(model)
    # print("model_jit: {}".format(model_jit.graph))

    # result = model(sem_logits, roi_msk_logits, bbx_pred, cls_pred)
    for idx in range(10):
        result = model(idx)
        print("result: {}".format(result))

        if idx == 0:
            torch.onnx.export(
            model=model, 
            args=(idx), 
            f="./trt_demo.onnx", 
            opset_version=13, verbose=True, do_constant_folding=True)

    test_onnx("./trt_demo.onnx")
