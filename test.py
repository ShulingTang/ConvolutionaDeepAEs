import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import math
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
    """

    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)

class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow.
    A tensor with width w_in, feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad:
        w_nopad = (w_in - 1) * stride + kernel
    If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding), the width of T_pad:
        w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding - output_padding)
    Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is actually deleting row/col.
    If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the left, i.e., the first ceil(pad/2) and
    last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad.
    In contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(pad/2)`
    columns are deleted.
    For the height, Pytorch deletes more rows at top, while Tensorflow at bottom.
    In practice, we usually want `w_pad = w_in * stride`, i.e., the "SAME" padding mode in Tensorflow,
    so the number of columns to delete:
        pad = 2*padding - output_padding = kernel - stride
    We can solve the above equation and get:
        padding = ceil((kernel - stride)/2), and
        output_padding = 2*padding - (kernel - stride) which is either 1 or 0.
    But to get the same result with Tensorflow, we should delete values by ourselves instead of using padding and
    output_padding in ConvTranspose2d.
    To get there, we check the following conditions:
    If pad = kernel - stride is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad = kernel - stride is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves; or we can use ConvTranspose2d to delete `pad - 1` by setting `padding=(pad - 1) / 2` and `ouput_padding=0`
    and then delete the last row/column of the resulting tensor by ourselves.
    Here we implement the former case.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    And this module can only output a tensor with shape `stride * size_input`.
    A more flexible module can be found in `yaleb.py` which can output arbitrary size as specified.
    """

    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module('conv%d' % i,
                                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module('deconv%d' % (i + 1),
                                    nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        x_const = self.decoder(h)
        return x_const

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = sio.loadmat('Mnist60000.mat')
    # print(data)
    imgs, labels = data['X'].reshape((-1, 1, 28, 28)), data['y']
    labels = np.squeeze(labels - 1)  # y in [0, 1, ..., K-1]

    # network and optimization parameters
    num_sample = imgs.shape[0]
    channels = [1, 16, 8]
    kernels = [3, 3]
    epoches = 40
    lr = 1e-2
    weight_decay = 1e-5
    
    if not isinstance(imgs, torch.Tensor):
        imgs = torch.tensor(imgs, dtype=torch.float32, device=device)
    imgs = imgs.to(device)
    if isinstance(labels, torch.Tensor):
        labels = labels.to('cpu').numpy()

    model = ConvAE(channels, kernels)
    if torch.cuda.is_available():
        model.to(device)

    optimizier = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epoches):
        if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizier.param_groups:
                param_group['lr'] *= 0.1
        for img in imgs:
            
            img = img.view(-1, img.size(0),img.size(1),img.size(2))
            # print(img.shape)
            img = img.to(device)
            # forward
            output = model(img)
            loss = criterion(output, img)
            # backward
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        print("epoch=", epoch, loss.data.float())
        for param_group in optimizier.param_groups:
            print(param_group['lr'])
        if (epoch+1) % 5 == 0:
            print("epoch: {}, loss is {}".format((epoch+1), loss.data))
            pic = to_img(output.cpu().data)
            if not os.path.exists('./simple_autoencoder'):
                os.mkdir('./simple_autoencoder')
            save_image(pic, './simple_autoencoder/image_{}.png'.format(epoch + 1))
    torch.save(model.state_dict(), './Mnist10000.pkl')
    # model = torch.load('./autoencoder.pkl')
    # test_img = imgs[20]
    # test_img = torch.tensor(test_img, dtype=torch.float32, device=device)
    # test_img = test_img.view(-1, test_img.size(0),test_img.size(1),test_img.size(2))
    # out_put = model(test_img)
    # reconst_img = to_img(out_put).squeeze()
    # reconst_img = reconst_img.data.cpu().data
    # plt.imshow(reconst_img.numpy().astype('uint8'), cmap='gray')
    # test_img = to_img(test_img).squeeze()
    # save_image(test_img, './simple_autoencoder/test_img.png')
    # save_image(reconst_img, './simple_autoencoder/reconst_img.png')
    # plt.show()