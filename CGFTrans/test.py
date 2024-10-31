import torch

# 创建原始三维张量
input_tensor = torch.tensor([[[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]],

                             [[13, 14, 15, 16],
                              [17, 18, 19, 20],
                              [21, 22, 23, 24]]])

# 删除第二维的最后一项数据
modified_tensor = torch.cat((input_tensor[:, :1, :], input_tensor[:, 1:, :]), dim=1)

print(input_tensor[:, :-1, :])