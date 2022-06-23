from pickle import NONE
import torch
import torch.nn.functional as F


class TrtDemo(torch.nn.Module):
    def __init__(self):
        super(TrtDemo, self).__init__()

    def forward(self, logits:torch.Tensor, indices:torch.Tensor):
        index = indices.view(-1, 1, 1, 1).expand(logits.shape)
        # results = torch.gather(logits, 1, index)
        results = logits.gather(1, index)
        return results

    def forward_3(self, logits:torch.Tensor, indices:torch.Tensor):
        N = logits.shape[0]
        results = torch.zeros(size=[1, 1, 28, 28], dtype=torch.float)
        # tmp = torch.zeros(size=[1, 1, 28, 28], dtype=torch.float)
        # results = [tmp]
        for idx in range(N):
            dim0 = torch.tensor([idx], dtype=torch.int64)
            dim1 = idx2 = torch.tensor(indices[idx], dtype=torch.int64)
            # this line is equivalent to "logits_i = logits[idx, indices[idx], :, :].unsqueeze(0).unsqueeze(0)"
            logits_i = torch.index_select(torch.index_select(logits, 0, dim0), 1, dim1)
            results = torch.cat([results, logits_i], dim=0)
            # results.append(logits_i)
            print("idx: {}, logits_i: {}, results: {}".format(idx, logits_i.shape, results.shape))
        return results

    def forward_2(self, sem_logits:torch.Tensor, roi_msk_logits:torch.Tensor, bbx:torch.Tensor, cls:torch.Tensor):
        # roi_msk_logits_i = roi_msk_logits
        cls_i = cls.squeeze(0)
        N = roi_msk_logits.shape[0]
        msk_logits = torch.zeros(size=[1, 1, 28, 28], dtype=torch.float)
        for idx in range(N):
            idx1 = torch.tensor([idx], dtype=torch.int64)
            idx2 = torch.tensor(cls_i[idx], dtype=torch.int64)
            # msk_logits_i = torch.flatten(msk_logits_i, start_dim=0, end_dim=1)
            roi_msk_logits_1 = torch.index_select(roi_msk_logits, 0, idx1)
            msk_logits_i = torch.index_select(roi_msk_logits_1, 1, idx2)
            print("idx: {}, msk_logits_i: {}, msk_logits: {}".format(idx, msk_logits_i.shape, msk_logits.shape))
            msk_logits = torch.cat((msk_logits, msk_logits_i), dim=1)
            # msk_logits = torch.add(msk_logits, msk_logits_i)
        return msk_logits

    # Internal Error ((Unnamed Layer* 15) [Recurrence]: inputs to IRecurrenceLayer have different dimensions. 
    # First input has dimensions [1,1,1,28,28] and second input has dimensions [1,1,2,28,28]
    def forward_1(self, sem_logits:torch.Tensor, roi_msk_logits:torch.Tensor, bbx:torch.Tensor, cls:torch.Tensor):
        # roi_msk_logits_i = roi_msk_logits
        cls_i = cls.squeeze(0)
        N = roi_msk_logits.shape[1]
        msk_logits = torch.zeros(size=[1, 1, 1, 28, 28], dtype=torch.float)
        for idx in range(N):
            # msk_logits_i = roi_msk_logits[:, idx, cls_i[idx], :, :]
            # msk_logits_i = torch.flatten(msk_logits_i, start_dim=0, end_dim=1)
            idx1 = torch.tensor([idx], dtype=torch.int64)
            idx2 = torch.tensor(cls_i[idx], dtype=torch.int64)
            roi_msk_logits_1 = torch.index_select(roi_msk_logits, 1, idx1)
            msk_logits_i = torch.index_select(roi_msk_logits_1, 2, idx2)
            print("idx: {}, msk_logits_i: {}, msk_logits: {}".format(idx, msk_logits_i.shape, msk_logits.shape))
            msk_logits = torch.cat((msk_logits, msk_logits_i), dim=2)
            # msk_logits = torch.add(msk_logits, msk_logits_i)
        return msk_logits

    # could not find plugin: SequenceEmpty
    def forward_0(self, sem_logits:torch.Tensor, roi_msk_logits:torch.Tensor, bbx:torch.Tensor, cls:torch.Tensor):
        # torch.zeros([1, 0, 28, 28]) lead-to "constant weights has count 0 but 1 was expected"
        msk_logits = torch.zeros([1, 1, 28, 28], dtype=torch.float, device=sem_logits.device)
        # msk_logits = []
        roi_msk_logits_i = roi_msk_logits[0]
        cls_i = cls[0]
        B = roi_msk_logits.shape[0]
        print("loop: {}".format(B))
        # for roi_msk_logits_i, cls_i in zip(roi_msk_logits, cls):
        for batch in range(B):
            # if len(roi_msk_logits_i) > 0:
            msk_logits_i = torch.cat([roi_msk_logits_i[idx, cls_i[idx], :, :].unsqueeze(0) for idx in range(roi_msk_logits_i.shape[0])], dim=0)
            print("batch: {}, msk_logits_i: {}, msk_logits: {}".format(batch, msk_logits_i.shape, msk_logits.shape))
            # msk_logits_i = torch.zeros(size=[1, 28, 28], dtype=torch.float)
            msk_logits = torch.cat((msk_logits, msk_logits_i.unsqueeze(0)), dim=1)
        # msk_logits = torch.stack(msk_logits)
        print("msk_logits: {}".format(msk_logits.shape))
        return msk_logits

if __name__ == "__main__":
    print("torch version: {}".format(torch.__version__))
    model = TrtDemo()
    model.eval()
    model_jit = torch.jit.script(model)
    # print("model_jit: {}".format(model_jit.graph))
    
    sem_logits = torch.rand(size=[1, 10, 896, 1536], dtype=torch.float)
    # roi_msk_logits = torch.rand(size=[1, 10, 4, 28, 28], dtype=torch.float)
    roi_msk_logits = torch.rand(size=[10, 4, 28, 28], dtype=torch.float)
    bbx_pred = torch.rand(size=[1, 10, 4], dtype=torch.float)
    cls_pred = (4*torch.rand([1, 10], dtype=torch.float)).type(torch.int64)

    logits = torch.rand(size=[10, 4, 28, 28], dtype=torch.float)
    indices = (torch.rand([10], dtype=torch.float)).type(torch.int64)

    # result = model(sem_logits, roi_msk_logits, bbx_pred, cls_pred)
    result = model(logits, indices)
    print("result: {}".format(result.shape))

    # after export, run onnx model with "polygraphy run --trt trt_demo.onnx"
    # torch.onnx.export(
    # model=model_jit, 
    # args=(sem_logits, roi_msk_logits, bbx_pred, cls_pred), 
    # f="./trt_demo.onnx", 
    # input_names=["sem_logits", "roi_msk_logits", "bbx_pred", "cls_pred"],
    # output_names=["po_pred_seamless"],
    # dynamic_axes={
    #         "roi_msk_logits": [1],
    #         "bbx_pred": [1],
    #         "cls_pred": [1],
    #         },
    # opset_version=13, verbose=True, do_constant_folding=True)

    torch.onnx.export(
    model=model_jit, 
    args=(logits, indices), 
    f="./trt_demo.onnx", 
    opset_version=13, verbose=True, do_constant_folding=True)
