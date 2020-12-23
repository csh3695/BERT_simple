import torch
from torch.utils.data import DataLoader


def trainer(model, optimizer, criteria, dataset, scheduler,
            batch_size=128, epoch=200, loss_list=[],
            mask_only=False, cuda=False):
    if cuda:
        model = model.cuda()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    lr_sch = scheduler
    for ep in range(epoch):
        optimizer.param_groups[0]['lr'] = next(lr_sch)
        for i, data in enumerate(dataloader):
            x, mask, y = data
            if cuda:
                x = x.cuda()
                y = y.cuda()
            y_pred = model(x)
            if mask_only:
                loss = criteria(y_pred[mask], y[mask])
            else:
                loss = criteria(y_pred.reshape(-1, y_pred.shape[-1]), y.reshape(-1))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if i % 4 == 0:
                print(f"EP\t{ep} | data\t{i} | loss\t{loss}")

    return {'ep': ep,
            'dat_cnt': i,
            'loss': loss_list
            }

