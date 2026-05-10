
import time, argparse, logging, sys, os, random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.getcwd())

from networks.bitm_arch import InverseToneMappingUNet
from networks.diffusion_reg import VPSDE, compute_pretrain_loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HDRDataset(Dataset):
    """Load HDR/clean images for unconditional diffusion pre-training."""
    EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.exr'}

    def __init__(self, root_dir, crop_size=256):
        self.crop_size = crop_size
        self.paths = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if os.path.splitext(f)[1].lower() in self.EXTS])
        assert len(self.paths) > 0, f'no images found in {root_dir}'

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        w, h = img.size
        cs = self.crop_size

        x = random.randint(0, max(w - cs, 0))
        y = random.randint(0, max(h - cs, 0))
        img = img.crop((x, y, x + cs, y + cs))

        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return transforms.ToTensor()(img)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# data
parser.add_argument('--hdr_dir', type=str, required=True)
parser.add_argument('--crop_size', type=int, default=256)

# training
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--T_period', type=int, default=50)
parser.add_argument('--grad_accum_steps', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--save_every', type=int, default=50)

# model
parser.add_argument('--base_channel', type=int, default=32)
parser.add_argument('--num_res', type=int, default=6)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[1, 1, 1, 28])
parser.add_argument('--dec_blks', nargs='+', type=int, default=[1, 1, 1, 1])
parser.add_argument('--kernel_size', type=int, default=3)

# diffusion
parser.add_argument('--diffusion_T', type=int, default=50)
parser.add_argument('--beta_min', type=float, default=0.1)
parser.add_argument('--beta_max', type=float, default=20.0)

# save
parser.add_argument('--save_path', type=str, default='./ckpt/')
parser.add_argument('--experiment_name', type=str, default='pretrain')
parser.add_argument('--writer_dir', type=str, default='./tf-logs/')

# resume
parser.add_argument('--resume', type=str, default='')

args = parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.writer_dir, exist_ok=True)

    writer = SummaryWriter(args.writer_dir + args.experiment_name)

    logging.basicConfig(
        filename=os.path.join(args.save_path, args.experiment_name + '.log'),
        level=logging.INFO)
    for k, v in vars(args).items():
        logging.info(f'{k}: {v}')

    # dataset
    dataset = HDRDataset(args.hdr_dir, crop_size=args.crop_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=8, drop_last=True)
    print(f'dataset: {len(dataset)} images, {len(dataloader)} batches')

    # model (same architecture as the restoration network)
    net = InverseToneMappingUNet(
        img_channel=3, width=args.base_channel,
        middle_blk_num=args.num_res,
        enc_blk_nums=args.enc_blks,
        dec_blk_nums=args.dec_blks,
        kernel_size=args.kernel_size,
    ).to(device)
    print(f'#parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M')

    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_period, T_mult=1)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f'resumed from epoch {start_epoch}')

    # training
    optimizer.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        net.train()
        st = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for i, hdr_imgs in enumerate(dataloader):
            hdr_imgs = hdr_imgs.to(device)

            loss = compute_pretrain_loss(net, hdr_imgs, sde, T=args.diffusion_T)
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (i + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum_steps
            num_batches += 1

            if (i + 1) % args.print_freq == 0:
                avg_loss = epoch_loss / num_batches
                print(f"epoch:{epoch} [{i+1}/{len(dataloader)}] "
                      f"lr:{optimizer.param_groups[0]['lr']:.7f} "
                      f"loss:{avg_loss:.5f} t:{time.time()-st:.1f}s")
                logging.info(f"epoch:{epoch} [{i+1}/{len(dataloader)}] loss:{avg_loss:.5f}")
                st = time.time()

        avg_loss = epoch_loss / max(num_batches, 1)
        writer.add_scalar('pretrain_loss', avg_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        print(f"epoch:{epoch} avg_loss:{avg_loss:.5f}")

        # save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            torch.save({
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args),
            }, os.path.join(args.save_path, 'latest_pretrain.pth'))
            torch.save(net.state_dict(),
                       os.path.join(args.save_path, 'pretrained_model.pth'))
            print(f'  -> checkpoint saved at epoch {epoch}')
            logging.info(f'checkpoint saved at epoch {epoch}')

    print('pre-training complete.')
