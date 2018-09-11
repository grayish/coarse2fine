import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from H36M import Human36m, Task
from demo import get_merged_image
from util import get_coord_3d, get_accuracy
from tensorboardX import SummaryWriter
import torch.nn as nn


# from visdom import Visdom

# viz = Visdom(env='result')
class Trainer(object):
    def __init__(self, model, optimizer, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.writer = SummaryWriter()

        cfg = self.config
        self.dataset = Human36m(
            task=Task.Train,
            augment=True,
            image_path=cfg.image_path,
            subjects=cfg.subjects,
            heatmap_xy_coefficient=cfg.heatmap_xy_coefficient,
            voxel_xy_resolution=cfg.voxel_xy_res,
            voxel_z_resolutions=cfg.voxel_z_res,
            joints=cfg.num_parts,
            num_split=cfg.num_split)
        self.loader = DataLoader(self.dataset, cfg.batch,
                                 shuffle=True, pin_memory=True, num_workers=cfg.workers)

    def train_step(self, epoch, step):
        cfg = self.config
        avg_loss, avg_acc = 0, 0
        swap_indices = torch.LongTensor(
            [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]).to(cfg.device)

        with tqdm(total=len(self.loader), unit=' iter', unit_scale=False) as progress:
            progress.set_description("train: epoch %d" % epoch)
            for images, voxels, raw_data in self.loader:
                outputs = self.model(images.to(cfg.device))

                losses = list()
                for idx, (out, vx) in enumerate(zip(outputs, voxels)):
                    vx = vx.to(cfg.device)
                    out = out.view(-1, cfg.num_parts, cfg.voxel_z_res[idx],
                                   cfg.voxel_xy_res, cfg.voxel_xy_res)
                    losses.append(self.criterion(out, vx))
                total_loss = sum(losses)

                self.optimizer.zero_grad()
                total_loss.backward()

                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                avg_loss += total_loss.item()

                # get accuracy
                with torch.no_grad():
                    # get fliped results
                    flip_imgs = images.flip(3).to(cfg.device)
                    flip_outputs = self.model(flip_imgs)

                # reflip result to original
                flip_out = flip_outputs[-1].view(-1, cfg.num_parts, cfg.voxel_z_res[-1],
                                                 cfg.voxel_xy_res, cfg.voxel_xy_res)
                flip_out = torch.index_select(flip_out, 1, swap_indices).flip(4)

                # get accuracy by adding two results
                out += flip_out
                acc = get_accuracy(out, voxels[-1].to(cfg.device), cfg)
                avg_acc += acc.item()

                self.writer.add_scalar('train/loss', total_loss.item(), step)
                self.writer.add_scalar('train/acc', acc.item(), step)

                if step % 100 == 0:
                    true_img = get_merged_image(voxels[-1], images.numpy())
                    infr_img = get_merged_image(out.div(2).cpu(), images.numpy())
                    self.writer.add_image('train/Ground truth image', true_img, step)
                    self.writer.add_image('train/Inference image', infr_img, step)

                step = step + 1
                progress.set_postfix(loss=total_loss.item(), acc=acc.item())
                progress.update(1)

        avg_loss /= len(self.loader)
        avg_acc /= len(self.loader)
        self.writer.add_scalar('train/avg_loss', avg_loss, step)
        self.writer.add_scalar('train/avg_acc', avg_acc, step)

        self.dataset.change_current_split_set()

        return step, avg_loss, avg_acc
