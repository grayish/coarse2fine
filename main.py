import os
import torch

from H36M import Task
from c2f_group import C2FGroupNorm
from hourglass import CoarseToFine
from log import LOGGER as L

from config import Config
from trainer import Trainer
from valid import validate


def main():
    cfg = Config(
        # model_fmt='old_{}.ckpt',
        model_fmt='batch_norm_{:03d}.ckpt',
        lr=2.5e-4,
        task=Task.Train)

    if not os.path.exists(cfg.ckpt_path):
        os.mkdir(cfg.ckpt_path)

    L.info('Current configurations ...')
    L.info(cfg)

    L.info('Searching for the pretrained model...')
    last_epoch = 0
    prev_model = None
    for _, _, files in os.walk(cfg.ckpt_path):
        for file in files:
            from parse import parse
            parsed = parse(cfg.model_fmt, file)
            if parsed is not None:
                _epoch = int(parsed[0])
                if _epoch > last_epoch:
                    last_epoch = _epoch

    if last_epoch > 0:
        _path = os.path.join(cfg.ckpt_path, cfg.model_fmt.format(last_epoch))
        prev_model = torch.load(_path)

        L.info('the pretrained model is found (%d epoch, %d step)... ' %
               (prev_model['epoch'], prev_model['step']))
    else:
        L.info('There are no pre-trained models..')

    L.info('Creating model...')
    # model = C2FGroupNorm(cfg.voxel_z_res, cfg.num_parts)
    model = CoarseToFine(cfg.voxel_z_res, cfg.num_parts)
    if prev_model is not None:
        L.info('Load state_dict of pre-trained model')
        model.load_state_dict(prev_model['state'])

    L.info('set torch device: %s' % cfg.device)
    model = model.to(cfg.device)

    if cfg.task == Task.Train:
        L.info('Start training sequence ...')
        L.info('Creating optimizer and criterion...')
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.learning_rate)
        criterion = torch.nn.MSELoss()
        if prev_model is not None:
            L.info('Load state_dict of pre-trained optimizer')
            optimizer.load_state_dict(prev_model['optimizer'])

        step = 0
        if prev_model is not None:
            step = prev_model['step']
        L.info('training start from global step %d' % step)

        torch.backends.cudnn.benchmark = True

        trainer = Trainer(model, optimizer, criterion, cfg)
        for epoch in range(last_epoch + 1, cfg.epoch):
            step, avg_loss, avg_acc = trainer.train_step(epoch, step)
            L.info('Epoch %d was done.' % epoch)
            L.info('average accuracy %f, average loss: %f' % (avg_acc, avg_loss))
            L.info('Saving the trained model... (%d epoch, %d step)' % (epoch, step))
            save_path = os.path.join(cfg.ckpt_path, cfg.model_fmt.format(epoch))
            torch.save({
                'config': cfg,
                'epoch': epoch,
                'step': step,
                'state': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_path)
            L.info('Done')
    elif cfg.task == Task.Valid:
        L.info('Start validation sequence ...')
        mpjpe = validate(model, cfg)
        L.info('total mpjpe: %f' % mpjpe)


if __name__ == "__main__":
    main()
