import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)



def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))

def NMAE(pred, true):
    return np.mean(np.abs(true - pred) / np.abs(true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

# import torch
# import numpy as np
# def RSE_torch(pred, true):
#     return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - torch.mean(true)) ** 2))
#
# def CORR_torch(pred, true):
#     u = ((true - torch.mean(true, dim=0)) * (pred - torch.mean(pred, dim=0))).sum(dim=0)
#     d = torch.sqrt(((true - torch.mean(true, dim=0)) ** 2 * (pred - torch.mean(pred, dim=0)) ** 2).sum(dim=0))
#     return torch.mean(u / (d + 1e-8))  # 加1e-8防止除0
#
# def MAE_torch(pred, true):
#     return torch.mean(torch.abs(true - pred))
#
# def MSE_torch(pred, true):
#     return torch.mean((true - pred) ** 2)
#
# def RMSE_torch(pred, true):
#     return torch.sqrt(MSE_torch(pred, true))
#
# def MAPE_torch(pred, true):
#     return torch.mean(torch.abs((true - pred) / (true + 1e-5)))  # 避免除0
#
# def MSPE_torch(pred, true):
#     return torch.mean(torch.square((true - pred) / (true + 1e-5)))  # 避免除0
#
# def metric(pred, true):
#     # 如果是 numpy，先转换为 Tensor
#     if isinstance(pred, np.ndarray):
#         pred = torch.tensor(pred, dtype=torch.float32).cuda()
#     if isinstance(true, np.ndarray):
#         true = torch.tensor(true, dtype=torch.float32).cuda()
#
#     mae = MAE_torch(pred, true)
#     mse = MSE_torch(pred, true)
#     rmse = RMSE_torch(pred, true)
#     mape = MAPE_torch(pred, true)
#     mspe = MSPE_torch(pred, true)
#
#     return mae.item(), mse.item(), rmse.item(), mape.item(), mspe.item()

