from data_provider.data_factory import data_provider, data_provider_fast
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.trend_tag import patch_analyze_trends_gpu, calculate_accuracy_gpu
from utils.kd import relation_kd, response_kd, attention_kd, contrastive_kd, causal_kd
from utils.img import temporal_contrastive_loss, enhanced_temporal_contrastive_loss
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args, args_img, args_weather):
        self.args_img = args_img  # 先设置args_img
        self.args_weather = args_weather
        super(Exp_Long_Term_Forecast, self).__init__(args)  # 再调用父类初始化
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.args_img).float()
        model_img = self.model_dict[self.args_img.model_img].Model(self.args, self.args_img, self.args_weather).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            model_img = nn.DataParallel(model_img, device_ids=self.args.device_ids)
        return model, model_img

    def _get_data(self, flag):
        # if self.args.data == 'Folsom':
            # data_provider = data_provider_fast
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_img_optim = optim.Adam(self.model_img.parameters(), lr=self.args_img.learning_rate)
        return model_optim, model_img_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_loss_img = []
        self.model.eval()
        self.model_img.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_img, batch_y_img, batch_x_weather, batch_y_weather) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_img = batch_x_img.float().to(self.device)
                batch_y_img = batch_y_img.float().to(self.device)
                
                batch_x_weather = batch_x_weather.float().to(self.device)
                batch_y_weather = batch_y_weather.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_img)
                        outputs_img = self.model_img(batch_x, batch_x_img)
                else:
                    outputs = self.model(batch_x, batch_x_img)
                    outputs_img = self.model_img(batch_x, batch_x_img, batch_y_img, batch_x_weather, batch_y_weather)
                    # outputs_img = self.model_img(batch_x, batch_x_img)
                
                if self.args.data == 'Folsom':
                    f_dim = -42 if self.args.features == 'MS' else 0
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, :]
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs_img = outputs_img[:, -self.args.pred_len:, :]
                outputs_img = outputs_img[:, :, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                pred_img = outputs_img.detach().cpu()
                true_img = batch_y.detach().cpu()  # 使用batch_y而不是batch_y_img

                loss = criterion(pred, true)
                loss_img = criterion(pred_img, true_img)

                total_loss.append(loss.item())
                total_loss_img.append(loss_img.item())
        total_loss = np.average(total_loss)
        total_loss_img = np.average(total_loss_img)

        self.model.train()
        self.model_img.train()
        return total_loss, total_loss_img

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping_img = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, model_img_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.kd_type == 'response':
            criterion_kd = response_kd()
        elif self.args.kd_type == 'relation':
            criterion_kd = relation_kd()
        elif self.args.kd_type == 'attention':
            criterion_kd = attention_kd()
        elif self.args.kd_type == 'contrastive':
            criterion_kd = contrastive_kd(contrast_type='temporal')
        elif self.args.kd_type == 'causal':
            criterion_kd = causal_kd(contrast_type='temporal')

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_img = []

            self.model.train()
            self.model_img.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_img, batch_y_img, batch_x_weather, batch_y_weather) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                model_img_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_img = batch_x_img.float().to(self.device)
                batch_y_img = batch_y_img.float().to(self.device)
                
                batch_x_weather = batch_x_weather.float().to(self.device)
                batch_y_weather = batch_y_weather.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs_img = self.model_img(batch_x_img, batch_x_mark, dec_inp, batch_y_mark)

                        if self.args.data == 'Folsom':
                            f_dim = -42 if self.args.features == 'MS' else 0
                        else:
                            f_dim = -1 if self.args.features == 'MS' else 0

                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        # projected_img_output = self.projection_layer(outputs_img)

                        loss_img = criterion(outputs_img, batch_y_img)
                        loss = criterion(outputs, batch_y)
                        # loss += criterion_kd(outputs, projected_img_output.detach()) * 0.01
                        train_loss.append(loss.item())
                        train_loss_img.append(loss_img.item())
                else:
                    if self.args.kd_type == 'attention':
                        outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs_img, attns_img = self.model_img(batch_x_img, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        # outputs, features = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_intermediates=True)
                        # outputs_img, features_img = self.model_img(batch_x_img, batch_x_mark, dec_inp, batch_y_mark, return_intermediates=True)
                        outputs = self.model(batch_x, batch_x_img)
                        # outputs_img = self.model_img(batch_x, batch_x_img)
                        outputs_img = self.model_img(batch_x, batch_x_img, batch_y_img, batch_x_weather, batch_y_weather)
                    


                    if self.args.data == 'Folsom':
                        f_dim = -42 if self.args.features == 'MS' else 0
                    else:
                        f_dim = -1 if self.args.features == 'MS' else 0

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
        
                    loss_img = criterion(outputs_img, batch_y)

                    # # knowledge distillation
                    # feat = features['encoder_features'][-1]
                    # feat_img = features_img['encoder_features'][-1]

                    # feat = feat.permute(0, 3, 2, 1)
                    # feat_img = feat_img.permute(0, 3, 2, 1)
                    # projected_img_feat = self.projection_layer(feat_img)
                    # del feat_img

                    # feat = feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2] * feat.shape[3])
                    # projected_img_feat = projected_img_feat.reshape(projected_img_feat.shape[0], projected_img_feat.shape[1], projected_img_feat.shape[2] * projected_img_feat.shape[3])
                    # feat = feat.permute(0, 2, 1)
                    # projected_img_feat = projected_img_feat.permute(0, 2, 1)

                    # projected_img_output = self.projection_layer(outputs_img)

                    if self.args.kd_type == 'response':
                        loss_distillation = criterion_kd(outputs, outputs_img.detach()) 
                    elif self.args.kd_type == 'relation':
                        loss_distillation = criterion_kd(outputs, outputs_img.detach()) 
                    elif self.args.kd_type == 'contrastive':
                        # loss_distillation = criterion_kd(outputs, projected_img_output.detach())
                        # loss_distillation = criterion_kd(feat, projected_img_feat.detach())
                        loss_distillation = 0
                    elif self.args.kd_type == 'causal':
                        # loss_distillation = criterion_kd(features[-1], features_img[-1].detach())
                        # loss_distillation = criterion_kd(feat, projected_img_feat.detach())
                        loss_distillation = 0
                    
                    loss += loss_distillation * self.args.kd_loss_weight

                    train_loss.append(loss.item())
                    train_loss_img.append(loss_img.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.scale(loss_img).backward() # 对img_loss也进行反向传播
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.model_img.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.step(model_img_optim) # 更新两个模型
                    scaler.update()
                else:
                    loss.backward(retain_graph=True)
                    loss_img.backward() # 对img_loss也进行反向传播
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.model_img.parameters(), max_norm=1.0)
                    model_optim.step()
                    model_img_optim.step() # 更新两个模型

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_img = np.average(train_loss_img)
            vali_loss, vali_loss_img = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_loss_img = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train img Loss: {3:.7f}  Vali Loss: {4:.7f} Vali img Loss: {5:.7f} Test Loss: {6:.7f} Test img Loss: {7:.7f}".format(
                epoch + 1, train_steps, train_loss, train_loss_img, vali_loss, vali_loss_img, test_loss, test_loss_img))
            
            # 修改：分别保存两个模型，使用各自的验证损失
            early_stopping(vali_loss, self.model, path, model_name='checkpoint.pth')
            early_stopping_img(vali_loss_img, self.model_img, path, model_name='checkpoint_img.pth')
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            adjust_learning_rate(model_img_optim, epoch + 1, self.args_img)

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        best_model_path_img = path + '/' + 'checkpoint_img.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.model_img.load_state_dict(torch.load(best_model_path_img))

        return self.model, self.model_img

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading models')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model_img.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint_img.pth')))

        preds = []
        trues = []
        preds_img = []
        trues_img = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_img, batch_y_img, batch_x_weather, batch_y_weather) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_img = batch_x_img.float().to(self.device)
                batch_y_img = batch_y_img.float().to(self.device)
                
                batch_x_weather = batch_x_weather.float().to(self.device)
                batch_y_weather = batch_y_weather.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_img)
                        outputs_img = self.model_img(batch_x, batch_x_img)
                else:
                    outputs = self.model(batch_x, batch_x_img)
                    # outputs_img = self.model_img(batch_x, batch_x_img)
                    outputs_img = self.model_img(batch_x, batch_x_img, batch_y_img, batch_x_weather, batch_y_weather)


                if self.args.data == 'Folsom':
                    f_dim = -42 if self.args.features == 'MS' else 0
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs_img = outputs_img[:, -self.args.pred_len:, :]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                outputs_img = outputs_img.detach().cpu().numpy()

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                outputs_img = outputs_img[:, :, f_dim:]

                pred = outputs
                pred_img = outputs_img
                true = batch_y
                true_img = batch_y  # 使用batch_y而不是batch_y_img

                preds.append(pred)
                trues.append(true)
                preds_img.append(pred_img)
                trues_img.append(true_img)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds_img = np.concatenate(preds_img, axis=0)
        trues_img = np.concatenate(trues_img, axis=0)
        print('test shape:', preds.shape, trues.shape, preds_img.shape, trues_img.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        preds_img = preds_img.reshape(-1, preds_img.shape[-2], preds_img.shape[-1])
        trues_img = trues_img.reshape(-1, trues_img.shape[-2], trues_img.shape[-1])
        print('test shape:', preds.shape, trues.shape, preds_img.shape, trues_img.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        mae_img, mse_img, rmse_img, mape_img, mspe_img = metric(preds_img, trues_img)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        print('mse_img:{}, mae_img:{}'.format(mse_img, mae_img))

        f = open("result_long_term_forecasting.txt", 'a')   
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        return

    def test_train(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='train')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results_train/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.data == 'Folsom':
                    f_dim = -42 if self.args.features == 'MS' else 0
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # if test_data.scale and self.args.inverse:
                #     shape = batch_y.shape
                #     if outputs.shape[-1] != batch_y.shape[-1]:
                #         outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                #     outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    # if test_data.scale and self.args.inverse:
                    #     shape = input.shape
                    #     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        return

    def test_trend(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.data == 'Folsom':
                    f_dim = -42 if self.args.features == 'MS' else 0
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # if test_data.scale and self.args.inverse:
                #     shape = batch_y.shape
                #     if outputs.shape[-1] != batch_y.shape[-1]:
                #         outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                #     outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)

        # preds_trend = patch_all(preds)
        # trues_trend = patch_all(trues)
        preds_trend = preds  # Placeholder
        trues_trend = trues  # Placeholder
        print(preds_trend)
        print(preds_trend.shape)
        print(trues_trend.shape)

        num_mismatches_np = np.sum(trues_trend != preds_trend)
        print((trues_trend.size-num_mismatches_np)/trues_trend.size)

        return
