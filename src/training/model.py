#

"""
I spent some good amount of time on making this model work as the models listed on SysCV/bdd100k-models
model zoo were not reachable. :sad:
"""
# import math
# import torch
# import torch.nn as nn
# import pathlib

# NUM_CLASSES = 10
# BDD_CLASSES = [
#     "bus", "car", "motor", "person", "rider",
#     "traffic light", "traffic sign", "train", "truck", "bike",
# ]


# class YOLOPMultiClass(nn.Module):
#     def __init__(self, num_classes=NUM_CLASSES, drop_seg=True):
#         super().__init__()
#         full = torch.hub.load("hustvl/yolop", "yolop", pretrained=True)
#         self.model = full.model
#         self.detector_index = full.detector_index
#         self.save           = full.save


#         for p in self.model.parameters():
#             p.requires_grad = False

#         # keep only the head free.
#         for conv in self._det().m:
#             for p in conv.parameters():
#                 p.requires_grad = True

#     def _det(self):
#         return list(self.model.children())[self.detector_index]


#     def forward(self, x):
#         """
#         this is modified forward function from yolop repo where we only do forward pass until we hit the detector index.
#         """
#         cache = []
#         for i, block in enumerate(self.model):
#             if block.from_ != -1:
#                 x = (cache[block.from_] if isinstance(block.from_, int)
#                      else [x if j == -1 else cache[j] for j in block.from_])
#             x = block(x)
#             if i == self.detector_index:
#                 return x
#             cache.append(x if block.index in self.save else None)

#     def unfreeze(self):
#         for p in self.model.parameters():
#             p.requires_grad = True

#     def trainable_params(self):
#         return [p for p in self.parameters() if p.requires_grad]


# def load_checkpoint(path:pathlib.Path, device="cpu"):
#     model = YOLOPMultiClass()
#     ckpt  = torch.load(path, map_location=device)
#     model.load_state_dict(ckpt["state_dict"])
#     return model.to(device)


# model = YOLOPMultiClass(10)

# #inference
# img = torch.randn(1,3,640,640)
# out = model(img)
# print(out[0].shape)
# print(out[1].shape)
# print(out[2].shape)
