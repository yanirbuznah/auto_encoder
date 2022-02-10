import math
import numbers
import os
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from IPython.display import Image
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class ImageDataset(Dataset):
    # torchvision.transforms.RandomErasing()
    def __init__(self, data_dir, mode='train', transforms=None, monet=False):
        images_dir = os.path.join(data_dir, 'monet_jpg' if monet else 'photo_jpg')

        # if mode == 'train':
        #     self.files_A = [os.path.join(A_dir, name) for name in sorted(os.listdir(A_dir))[:250]]
        #     self.files_B = [os.path.join(B_dir, name) for name in sorted(os.listdir(B_dir))[:250]]
        # elif mode == 'test':
        #     self.files_A = [os.path.join(A_dir, name) for name in sorted(os.listdir(A_dir))[250:]]
        #     self.files_B = [os.path.join(B_dir, name) for name in sorted(os.listdir(B_dir))[250:301]]

        self.files = [os.path.join(images_dir, name) for name in sorted(os.listdir(images_dir))]
        # self.files_B = [os.path.join(B_dir, name) for name in sorted(os.listdir(B_dir))]

        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        # file_B = self.files_B[index]

        img = Image.open(file)
        # img_B = Image.open(file_B)

        if self.transforms is not None:
            original_img = self.transforms[0](img)

            img_with_erased_region = self.transforms[1](original_img)
            erased_region = 1 - abs(img_with_erased_region - original_img)
            # img_B = self.transforms(img_B)

            return original_img, erased_region, img_with_erased_region
        return img, img, img


class AutoEncoder(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super().__init__()
        if encoder is None or decoder is None:
            self.decoder, self.encoder = auto_encoder_parameters()
        else:
            self.encoder = encoder
            self.decoder = decoder

    def forward(self, x):
        encoder_output = self.encoder(x)
        output = self.decoder(encoder_output)
        return output


class Discriminator(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.model = discriminator_parameters() if model is None else model

    def forward(self, x):
        output = self.model(x)
        return output


class RandomBlocksErasing:

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, center=True, inplace=False,
                 random_blocks=0):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
        self.center = center
        self.random_blocks = random_blocks

    @staticmethod
    def get_params(img, scale, ratio, value=0, center=True):
        """Get parameters for ``erase`` for a random erasing.
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.
        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                if center:
                    i = (img_h - h) // 2
                    j = (img_w - w) // 2
                else:
                    i = random.randint(0, img_h - h)
                    j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                else:
                    v = None
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
        Returns:
            img (Tensor): Erased Tensor image.
        """
        if self.random_blocks > 0:
            num_of_blocks = random.randint(1, self.random_blocks)
            for i in range(num_of_blocks):
                if random.uniform(0, 1) < self.p:
                    x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value,
                                                    center=False)
                    img = F.erase(img, x, y, h, w, v, self.inplace)
            return img
        else:
            if random.uniform(0, 1) < self.p:
                x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value,
                                                center=self.center)
                return F.erase(img, x, y, h, w, v, self.inplace)
            return img


class RandomRegionErasing:
    def __init__(self, p=0.5, region_size=100, inplace=False):
        self.p = p
        self.region_size = region_size
        self.inplace = inplace

    @staticmethod
    def get_random_mask(img, region_size):
        img_c, img_h, img_w = img.shape

        n = 10
        mask = np.zeros((region_size, region_size))
        generator = np.random.RandomState()
        points = region_size * generator.rand(2, n ** 2)
        mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
        mask = ndimage.gaussian_filter(mask, sigma=region_size / (4. * n))
        mask = (mask > mask.mean()).astype(np.float)
        img = np.ones((img_h, img_w))
        start_h, start_w = random.randint(0, img_w - region_size), random.randint(0, img_w - region_size)

        img[start_h:start_h + region_size, start_w:start_w + region_size] = mask
        return img

        # mask = torch.ones(img_h, img_w)
        # start_h, start_w = random.randint(0, img_w - region_size), random.randint(0, img_w - region_size)
        # for i in range(region_size):
        #
        #     for j in range(region_size):
        #         r = 0.3
        #         r += 0.25 if mask[start_h + i, start_w + j -1] == 0 else 0
        #         r += 0.25 if mask[start_h + i -1, start_w + j] == 0 else 0
        #         if random.uniform(0, 1) < r:
        #             mask[start_h + i, start_w + j] = 0
        #         # if random.uniform(0, 1) < 0.1:
        #         #     old_j = j
        #         #     j = random.randint(j,region_w)
        #         #     mask[start_h + i, start_w + old_j:start_w + j] = 0
        #
        # return mask

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
        Returns:
            img (Tensor): Erased Tensor image.
        """
        if random.uniform(0, 1) < self.p:
            mask = self.get_random_mask(img, region_size=self.region_size)

            if not self.inplace:
                img = img.clone()
            mask_3d = mask[None, :, :] * np.ones(3, dtype=int)[:, None, None]
            indices_mask = np.where(mask_3d == 0)
            img[indices_mask] = 1

        return img


def central_region_transformer(p=1, scale=(0.0625, 0.0625), value=1, ratio=(1, 1)):
    return torchvision.transforms.Compose([
        RandomBlocksErasing(p, scale, ratio, value, center=True, random_blocks=0),
    ])


def random_blocks_transformer(p=1, scale=(0.02, 0.02), value=1, ratio=(1, 1)):
    return torchvision.transforms.Compose([
        RandomBlocksErasing(p, scale, ratio, value, center=False, random_blocks=10),

    ])


def random_region_transformer(p=1, region_size=100):
    return torchvision.transforms.Compose([
        RandomRegionErasing(p, region_size=region_size),
    ])


def general_transformer():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


def crop_image(batch):
    batch_size, dim, img_w, img_h = batch.shape
    i = (img_h - 64) // 2
    j = (img_w - 64) // 2
    croped_batch = batch[:, :, i:i + 64, j:j + 64]
    return croped_batch


def save_weights(model, path):
    torch.save(model.state_dict(), path)


def load_weights(model, path):
    model.load_state_dict(torch.load(path))


def add_missing_region_to_image(img, region):
    dim, img_w, img_h = img.shape
    i = (img_h - 64) // 2
    j = (img_w - 64) // 2
    img[:, i:i + 64, j:j + 64] = region
    return img


def train(dataloader, auto_encoder_pack, discriminator_pack):
    auto_encoder, ae_criterion, ae_optimizer, a = auto_encoder_pack['model'], auto_encoder_pack['loss'], \
                                                  auto_encoder_pack[
                                                      'optimizer'], auto_encoder_pack['lambda']
    discriminator, discriminator_criterion, discriminator_optimizer = discriminator_pack['model'], discriminator_pack[
        'loss'], discriminator_pack['optimizer']
    real_label, fake_label = 1, 0
    size = len(dataloader.dataset)
    auto_encoder.train()
    discriminator.train()
    train_ae_loss, train_d_loss, correct = 0, 0, 0
    for batch, X in enumerate(dataloader):
        original_img, erased_region, img_with_erased_region = X
        X = img_with_erased_region.to(device)
        real = crop_image(erased_region).to(device)

        # discriminator phase
        discriminator.zero_grad()
        b_size = real.size(0)
        r_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        f_label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        r_output = discriminator(real).view(-1)

        correct += (r_output > 0.5).type(torch.float).sum().item()
        # Calculate loss on all-real batch
        errD_real = discriminator_criterion(r_output, r_label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = r_output.mean().item()

        # Compute prediction error
        auto_encoder.zero_grad()
        fake = auto_encoder(X)
        # label.fill_(fake_label)
        # Classify all fake batch with D
        f_output = discriminator(fake.detach()).view(-1)
        correct += (f_output < 0.5).type(torch.float).sum().item()
        # Calculate D's loss on the all-fake batch
        errD_fake = discriminator_criterion(f_output, f_label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = f_output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake

        # Update D
        discriminator_optimizer.step()

        errAE = ae_criterion(fake, real)

        f_output = discriminator(fake.detach()).view(-1)

        r_output = discriminator(real).view(-1)
        adversarial_loss = discriminator_criterion(f_output, r_label)  # + discriminator_criterion(r_output,r_label)
        loss = a * errAE + (1 - a) * (adversarial_loss)

        train_ae_loss += loss.item()
        train_d_loss += errD.item()

        # Backpropagation
        ae_optimizer.zero_grad()
        loss.backward()
        a.backward()
        ae_optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_ae_loss /= len(dataloader)
    train_d_loss /= len(dataloader)
    correct *= (50 / size)
    index = random.randint(0, real.shape[0] - 1)
    pred = add_missing_region_to_image(img_with_erased_region[index], fake[index])
    pred = pred.permute(1, 2, 0).cpu().detach().numpy()
    y = original_img[index].permute(1, 2, 0).cpu()

    print(
        f"Train Error: \nAvg auto encoder loss: {train_ae_loss:>8f} Avg discriminator loss: {train_d_loss:>8f} acc:{correct:>8f}")
    return train_ae_loss, pred, y


def validate(dataloader, model, loss_fn, check_best=False):
    global best
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    validate_loss, correct = 0, 0
    with torch.no_grad():
        for X in dataloader:
            original_img, erased_region, img_with_erased_region = X
            X = img_with_erased_region.to(device)
            y = crop_image(erased_region).to(device)
            pred = model(X)
            validate_loss += loss_fn(pred, y).item()

    validate_loss /= num_batches

    #    if check_best and correct > best:
    # save_weights(model, 'weights.bin')
    index = random.randint(0, y.shape[0] - 1)
    pred = add_missing_region_to_image(img_with_erased_region[index], pred[index])
    pred = pred.permute(1, 2, 0).cpu().detach().numpy()
    y = original_img[index].permute(1, 2, 0).cpu()

    print(f"Validate Error: \nAvg loss: {validate_loss:>8f} \n")
    return validate_loss, pred, y


def train_ae_only(dataloader, auto_encoder_pack):
    auto_encoder, ae_criterion, ae_optimizer, a = auto_encoder_pack['model'], auto_encoder_pack['loss'], \
                                                  auto_encoder_pack[
                                                      'optimizer'], auto_encoder_pack['lambda']
    size = len(dataloader.dataset)
    auto_encoder.train()
    train_ae_loss = 0
    for batch, X in enumerate(dataloader):
        original_img, erased_region, img_with_erased_region = X
        X = img_with_erased_region.to(device)
        real = crop_image(erased_region).to(device)

        # Compute prediction error
        auto_encoder.zero_grad()
        fake = auto_encoder(X)

        loss = ae_criterion(fake, real)

        train_ae_loss += loss.item()

        # Backpropagation
        ae_optimizer.zero_grad()
        loss.backward()
        a.backward()
        ae_optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_ae_loss /= len(dataloader)

    index = random.randint(0, real.shape[0] - 1)
    pred = add_missing_region_to_image(img_with_erased_region[index], fake[index])
    pred = pred.permute(1, 2, 0).cpu().detach().numpy()
    y = original_img[index].permute(1, 2, 0).cpu()

    print(f"Train Error: \nAvg auto encoder loss: {train_ae_loss:>8f}")
    return train_ae_loss, pred, y


def main_loop(auto_encoder_pack, discriminator_pack, train_loader, validate_loader):
    epochs = 50
    # load_weights(discriminator_pack['model'], 'd_weights.bin')
    for t in range(10):
        print(f"Epoch {t + 1}\n-------------------------------")
        loss, train_pred, train_y = train_ae_only(dataloader=train_loader, auto_encoder_pack=auto_encoder_pack)
        loss, val_pred, val_y = validate(dataloader=validate_loader, model=auto_encoder_pack['model'],
                                         loss_fn=nn.MSELoss())
        show_images(train_y, train_pred, val_y, val_pred)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        loss, train_pred, train_y = train(dataloader=train_loader, auto_encoder_pack=auto_encoder_pack,
                                          discriminator_pack=discriminator_pack)
        loss, val_pred, val_y = validate(dataloader=validate_loader, model=auto_encoder_pack['model'],
                                         loss_fn=nn.MSELoss())
        show_images(train_y, train_pred, val_y, val_pred)

        # train_loss.append(loss)
        # train_correct.append(correct)
        # loss, correct = validate(validate_loader, model, nn.CrossEntropyLoss(), check_best=True)
        # validate_loss.append(loss)
        # validate_correct.append(correct)
    save_weights(auto_encoder_pack['model'], 'ae_weights.bin')
    save_weights(discriminator_pack['model'], 'd_weights.bin')
    print("Done!")


def show_images(train_true, train_fake, validate_true, validate_fake):
    # train_true = (1+train_true)/2
    # train_fake = (1+train_fake)/2
    # validate_true = (1+validate_true)/2
    # validate_fake = (1+validate_fake)/2
    fig, axs = plt.subplots(2, 2)
    fig.set_dpi(150)
    axs[0, 0].imshow(train_true, aspect='auto')
    axs[0, 0].set(ylabel='true')
    axs[0, 0].axis('off')
    axs[1, 0].imshow(train_fake, aspect='auto')
    axs[1, 0].set(xlabel='train', ylabel='fake')
    axs[1, 0].axis('off')
    axs[0, 1].imshow(validate_true, aspect='auto')
    axs[0, 1].axis('off')
    axs[1, 1].imshow(validate_fake, aspect='auto')
    axs[1, 1].set(xlabel='validate')
    axs[1, 1].axis('off')
    axs[0, 0].set_title('Train')
    axs[0, 1].set_title('Validate')
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()


def show_sample(data_set):
    img_a, img_b, img_c = data_set[0]
    plt.imshow(img_a.permute(1, 2, 0))
    plt.show()
    plt.imshow(img_b.permute(1, 2, 0))
    plt.show()
    plt.imshow(img_c.permute(1, 2, 0))
    plt.show()


def get_data_from_files():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './'
    data_set = ImageDataset(data_dir=data_dir, transforms=[general_transformer(), central_region_transformer()])
    train_size = int(0.80 * len(data_set))
    validate_size = len(data_set) - train_size
    train_dataset, validate_dataset = torch.utils.data.random_split(data_set, [train_size, validate_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=64, shuffle=True)
    return data_set, train_loader, validate_loader


def auto_encoder_parameters():
    encoder = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(4, 4), stride=(2, 2), padding=1),  # output: 32,128,128
        nn.BatchNorm2d(32),
        nn.PReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=1),  # output: 64,64,64
        nn.BatchNorm2d(64),
        nn.PReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=1),  # output: 128,32,32
        nn.BatchNorm2d(128),
        nn.PReLU(),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=1),  # output: 256,16,16
        nn.BatchNorm2d(256),
        nn.PReLU(),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=1),  # output: 512,8,8
        nn.BatchNorm2d(512),
        nn.PReLU(),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(4, 4), stride=(2, 2), padding=1),  # output: 1024,4,4
        nn.BatchNorm2d(1024),
        nn.PReLU(),
        nn.Conv2d(in_channels=1024, out_channels=8000, kernel_size=(4, 4), stride=(2, 2), padding=0),
        # output: 4000,1,1
        nn.BatchNorm2d(8000),
        nn.PReLU(),
    )
    decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels=8000, out_channels=1024, kernel_size=(4, 4), stride=(2, 2)),  # output: 1024,4,4
        nn.BatchNorm2d(1024),
        nn.PReLU(),
        nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        # output: 512,8,8
        nn.BatchNorm2d(512),
        nn.PReLU(),
        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        # output: 256,16,16
        nn.BatchNorm2d(256),
        nn.PReLU(),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        # output: 128,32,32
        nn.BatchNorm2d(128),
        nn.PReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        # output: 64,64,64
        nn.BatchNorm2d(64),
        nn.PReLU(),
        nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(1, 1)),  # output: 3,64,64
        nn.Tanh(),
        # nn.BatchNorm2d(3),
    )

    return decoder, encoder


def discriminator_parameters():
    discriminator = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.BatchNorm2d(64),
        nn.PReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.BatchNorm2d(128),
        nn.PReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.BatchNorm2d(256),
        nn.PReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.BatchNorm2d(512),
        nn.PReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.BatchNorm2d(1024),
        nn.PReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(2, 2), stride=(1, 1)),
        nn.Sigmoid(),

    )
    return discriminator


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    data_set, train_loader, validate_loader = get_data_from_files()
    # show_sample(data_set)
    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))
    discriminator_pack = {'model': discriminator, 'loss': nn.BCELoss(), 'optimizer': optimizerD}
    auto_encoder = AutoEncoder().to(device)
    auto_encoder.apply(weights_init)
    loss_lambda = nn.Parameter(torch.tensor(0.5))
    optimizerAE = torch.optim.Adam(list(auto_encoder.parameters()), lr=0.0003, betas=(0.5, 0.999))
    auto_encoder_pack = {'model': auto_encoder, 'loss': nn.MSELoss(), 'optimizer': optimizerAE, 'lambda': loss_lambda}

    main_loop(auto_encoder_pack, discriminator_pack, train_loader, validate_loader)


if __name__ == '__main__':
    main()
