from src.Warehouse import *
from src.tools import EarlyStopping

def distillation_loss(student_output, teacher_output, T=2.0):

    student_output = student_output.squeeze(-1)
    p = F.log_softmax(student_output / T, dim=-1)
    q = F.softmax(teacher_output / T, dim=-1)
    return F.kl_div(p, q, reduction='batchmean') * (T ** 2)

def vali(epoch,vali_data, vali_loader, criterion, pred_len,label_len,model,device,draw):
    total_loss = []
    total_mae_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -pred_len:, :]
            batch_y = batch_y[:, -pred_len:, -1:].to(device)
            # if draw == 1 and (epoch+1)%5 == 0:
            #     Curve(epoch=epoch, prediction=outputs, y_test=batch_y)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            mae = nn.L1Loss()

            loss = criterion(pred, true)
            mae_loss = mae(pred, true)

            total_loss.append(loss)
            total_mae_loss.append(mae_loss)

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss


def exper(cell, model_name,NUMEPOCH, student_model, teacher_model, train_loader, model_optim, criterion, vali_data, vali_loader,
          test_data, test_loader, pred_len, label_len, device):
    path = os.path.join(f'./checkpoints/cell{cell}', 'student_checkpoint')
    if not os.path.exists(path):
        os.makedirs(path)
    alpha = 0.3
    early_stopping = EarlyStopping(model_name = model_name,patience=10, verbose=True)
    time_now = time.time()
    train_steps = len(train_loader)

    results = {'train_loss': [], 'valid_loss': [], 'test_loss': [], 'test_mae_loss': [],'cost_time': []}
    if not os.path.exists(f'checkpoints/cell{cell}/loss_result'):
        os.mkdir(f'checkpoints/cell{cell}/loss_result')

    for epoch in range(1, NUMEPOCH + 1):
        iter_count = 0
        train_loss = []

        student_model.train().cuda()
        teacher_model.eval().cuda()
        epoch_time = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

            student_outputs = student_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            student_outputs = student_outputs[:, -pred_len:, :]

            teaher_x = batch_x.squeeze(-1)
            teacher_outputs = teacher_model(input_ids=teaher_x).logits

            batch_y = batch_y[:, -pred_len:, -1:].to(device)

            loss = alpha * distillation_loss(student_outputs, teacher_outputs) + (1 - alpha) * criterion(
                student_outputs, batch_y)
            train_loss.append(loss.item())

            loss.backward()
            model_optim.step()

            # 更新 tqdm 的描述信息
            pbar.set_postfix(loss=loss.item())

        results['cost_time'].append(time.time() - epoch_time)
        tqdm.write(f"Epoch: {epoch} cost time: {time.time() - epoch_time:.2f}s")

        train_loss = np.average(train_loss)
        vali_loss, _ = vali(epoch, vali_data, vali_loader, criterion, pred_len, label_len, student_model, device,
                            draw=0)
        test_loss, test_mae_loss = vali(epoch, test_data, test_loader, criterion, pred_len, label_len, student_model,
                                        device, draw=1)

        results['train_loss'].append(train_loss)
        results['valid_loss'].append(vali_loss)
        results['test_loss'].append(test_loss)
        results['test_mae_loss'].append(test_mae_loss)

        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('checkpoints/cell{}/loss_result/{}_statistics.csv'.format(cell,model_name), index_label='epoch')

        tqdm.write("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch, train_steps, train_loss, vali_loss, test_loss))

        early_stopping(vali_loss, student_model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_model_path = os.path.join(path + '/' + '{}_checkpoint.pth'.format(model_name))
    torch.save(student_model.state_dict(), best_model_path)
    return student_model, early_stopping.early_stop