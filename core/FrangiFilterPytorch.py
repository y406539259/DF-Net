import numpy as np
import cv2
import math
from scipy import signal
import torch
import torch.utils.data
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple

# modify con2d function to use same padding
# code referd to @famssa in 'https://github.com/pytorch/pytorch/issues/3867'
# and tensorflow source code

class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# custom con2d, because pytorch don't have "padding='same'" option.

def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)

    cols_odd = (padding_rows % 2 != 0)
    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)

def eig2imagePytorch(Dxx1, Dxy1, Dyy1):
    # This function eig2image calculates the eigen values from the
    # hessian matrix, sorted by abs value. And gives the direction
    # of the ridge (eigenvector smallest eigenvalue) .
    # output:Lambda1,Lambda2,Ix,Iy
    # Compute the eigenvectors of J, v1 and v2
    tmp1 = torch.sqrt(torch.pow(Dxx1 - Dyy1 ,2) + 4 *torch.pow(Dxy1,2))

    v2x1 = 2 * Dxy1
    v2y1 = Dyy1 - Dxx1 + tmp1

    mag1 = torch.sqrt(torch.pow(v2x1, 2) + torch.pow(v2y1, 2))
    i1 = mag1 != 0

    v2x1[i1 == 1] = v2x1[i1 == 1] / mag1[i1 == 1]
    v2y1[i1 == 1] = v2y1[i1 == 1] / mag1[i1 == 1]

    v1x1 = -v2y1
    v1y1 = v2x1

    mu11 = 0.5 * (Dxx1 + Dyy1 + tmp1)
    mu21 = 0.5 * (Dxx1 + Dyy1 - tmp1)

    check1 = torch.abs(mu11) > torch.abs(mu21)

    Lambda11 = mu11.clone()
    Lambda11[check1 == 1] = mu21[check1 == 1]
    Lambda21 = mu21
    Lambda21[check1 == 1] = mu11[check1 == 1]

    Ix1 = v1x1
    Ix1[check1 == 1] = v2x1[check1 == 1]
    Iy1 = v1y1
    Iy1[check1 == 1] = v2y1[check1 == 1]

    return Lambda11, Lambda21, Ix1, Iy1

def Hessian2DPytorch(I, Sigma, S_round):
    #I = np.array(I, dtype=float)
    #I = torch.from_numpy(I).double()
    #I = torch.unsqueeze(I, 0)
    #I = torch.unsqueeze(I, 0)

    Sigma = np.array(Sigma, dtype=float)
    #S_round = np.round(3 * Sigma)

    [X, Y] = np.mgrid[-S_round:S_round + 1, -S_round:S_round + 1]

    DGaussxx = 1 / (2 * math.pi * pow(Sigma, 4)) * (X ** 2 / pow(Sigma, 2) - 1) * np.exp(
        -(X ** 2 + Y ** 2) / (2 * pow(Sigma, 2)))
    DGaussxy = 1 / (2 * math.pi * pow(Sigma, 6)) * (X * Y) * np.exp(-(X ** 2 + Y ** 2) / (2 * pow(Sigma, 2)))
    DGaussyy = 1 / (2 * math.pi * pow(Sigma, 4)) * (Y ** 2 / pow(Sigma, 2) - 1) * np.exp(
        -(X ** 2 + Y ** 2) / (2 * pow(Sigma, 2)))

    Dif_xx = Conv2d(in_channels=1, out_channels=1, kernel_size=(2 * S_round + 1, 2 * S_round + 1), bias=False, padding=1)
    Dif_xx_weight = DGaussxx
    Dif_yy = Conv2d(in_channels=1, out_channels=1, kernel_size=(2 * S_round + 1, 2 * S_round + 1), bias=False, padding=1)
    Dif_yy_weight = DGaussyy
    Dif_xy = Conv2d(in_channels=1, out_channels=1, kernel_size=(2 * S_round + 1, 2 * S_round + 1), bias=False, padding=1)
    Dif_xy_weight = DGaussxy

    Dif_xx_weight = Dif_xx_weight[np.newaxis, np.newaxis, :, :]
    Dif_yy_weight = Dif_yy_weight[np.newaxis, np.newaxis, :, :]
    Dif_xy_weight = Dif_xy_weight[np.newaxis, np.newaxis, :, :]

    Dif_xx_weight = torch.DoubleTensor(Dif_xx_weight).cuda()
    Dif_yy_weight = torch.DoubleTensor(Dif_yy_weight).cuda()
    Dif_xy_weight = torch.DoubleTensor(Dif_xy_weight).cuda()

    Dif_xx.weight.data = Dif_xx_weight
    Dif_yy.weight.data = Dif_yy_weight
    Dif_xy.weight.data = Dif_xy_weight

    Dif_xx.weight.requires_grad = False
    Dif_yy.weight.requires_grad = False
    Dif_xy.weight.requires_grad = False

    Dxx = Dif_xx(I)
    Dxy = Dif_xy(I)
    Dyy = Dif_yy(I)
    #print(Dxx.size())
    #print(Dxy.size())
    #print(Dyy.size())

    return Dxx, Dxy, Dyy

def FrangiFilter2DPytorch(I):
    I = np.array(I, dtype=float)
    defaultoptions = {'FrangiScaleRange': (1, 10), 'FrangiScaleRatio': 2, 'FrangiBetaOne': 0.5, 'FrangiBetaTwo': 15,
                      'verbose': False, 'BlackWhite': False};
    options = defaultoptions
    sigmas = np.arange(options['FrangiScaleRange'][0], options['FrangiScaleRange'][1], options['FrangiScaleRatio'])
    sigmas.sort()

    beta = 2 * pow(options['FrangiBetaOne'], 2)
    c = 2 * pow(options['FrangiBetaTwo'], 2)

    shape = (I.shape[0], I.shape[1], len(sigmas))
    ALLfiltered =[]
    ALLangles = np.zeros(shape)

    # Frangi filter for all sigmas
    Rb = 0
    S2 = 0
    count = 0
    for i in range(len(sigmas)):
        # Show progress
        #if (options['verbose']):
        #    print('Current Frangi Filter Sigma: ', sigmas[i])

        # Make 2D hessian
        Dxx, Dxy, Dyy = Hessian2DPytorch(I, sigmas[i], 3 * sigmas[i])

        # Correct for scale
        Dxx = pow(sigmas[i], 2) * Dxx
        Dxy = pow(sigmas[i], 2) * Dxy
        Dyy = pow(sigmas[i], 2) * Dyy

        # Calculate (abs sorted) eigenvalues and vectors
        [Lambda2, Lambda1, Ix, Iy] = eig2imagePytorch(Dxx, Dxy, Dyy)

        # Compute the direction of the minor eigenvector
        angles = torch.atan2(Ix, Iy)

        # Compute some similarity measures
        Lambda1[Lambda1 == 0] = np.spacing(1)

        Rb = (Lambda2 / Lambda1) ** 2
        S2 = Lambda1 ** 2 + Lambda2 ** 2

        # Compute the output image
        Ifiltered = torch.exp(-Rb / beta) * (np.ones(I.shape) - torch.exp(-S2 / c))

        # see pp. 45
        if (options['BlackWhite']):
            Ifiltered[Lambda1 < 0] = 0
        else:
            Ifiltered[Lambda1 > 0] = 0

        ALLfiltered.append(Ifiltered)
        # Return for every pixel the value of the scale(sigma) with the maximum
        # output pixel value

    if len(sigmas) > 1:
        ALLfiltered = torch.stack(ALLfiltered)
        outIm, outImIndex  = torch.max(ALLfiltered, dim=0)
    else:
        outIm = (outIm.transpose()).reshape(I.shape)

    return outIm


def FrangiFilter2DPytorchSmall(I):
    defaultoptions = {'FrangiScaleRange': (1, 5), 'FrangiScaleRatio':1, 'FrangiBetaOne': 0.5, 'FrangiBetaTwo': 15,
                      'verbose': False, 'BlackWhite': True};
    options = defaultoptions
    sigmas = np.arange(options['FrangiScaleRange'][0], options['FrangiScaleRange'][1], options['FrangiScaleRatio'])
    sigmas.sort()

    beta = 2 * pow(options['FrangiBetaOne'], 2)
    c = 2 * pow(options['FrangiBetaTwo'], 2)
    #beta = 1
    #c = 450

    shape = (I.size()[1], I.size()[2], len(sigmas))
    ALLfiltered =[]
    #ALLangles = np.zeros(shape)

    # Frangi filter for all sigmas
    Rb = 0
    S2 = 0
    count = 0
    for i in range(len(sigmas)):
        # Show progress
        if (options['verbose']):
            print('Current Frangi Filter Sigma: ', sigmas[i])

        # Make 2D hessian
        Dxx, Dxy, Dyy = Hessian2DPytorch(I, sigmas[i], 3 * sigmas[i])
        #Dxx, Dxy, Dyy = Hessian2DPytorch(I, sigmas[i], 3)

        # Correct for scale
        Dxx = pow(sigmas[i], 2) * Dxx
        Dxy = pow(sigmas[i], 2) * Dxy
        Dyy = pow(sigmas[i], 2) * Dyy

        # Calculate (abs sorted) eigenvalues and vectors
        [Lambda2, Lambda1, Ix, Iy] = eig2imagePytorch(Dxx, Dxy, Dyy)

        # Compute the direction of the minor eigenvector
        angles = torch.atan2(Ix, Iy)

        # Compute some similarity measures
        Lambda1[Lambda1 == 0] = np.spacing(1)

        Rb = (Lambda2 / Lambda1) ** 2
        S2 = Lambda1 ** 2 + Lambda2 ** 2


        #a = torch.exp(-Rb / beta)
        #b = torch.ones(I.size()).double()
        # Compute the output image
        Ifiltered = torch.exp(-Rb / beta) * (torch.ones(I.size()).double().cuda() - torch.exp(-S2 / c))

        # see pp. 45
        if (options['BlackWhite']):
            Ifiltered[Lambda1 < 0] = 0
        else:
            Ifiltered[Lambda1 > 0] = 0

        ALLfiltered.append(Ifiltered)
        # Return for every pixel the value of the scale(sigma) with the maximum
        # output pixel value

    if len(sigmas) > 1:
        ALLfiltered = torch.stack(ALLfiltered)
        outIm, outImIndex  = torch.max(ALLfiltered, dim=0)
    else:
        outIm = (outIm.transpose()).reshape(I.shape)

    #outIm = torch.squeeze(outIm)
    #outIm = outIm.cpu().data.numpy()
    #min = np.min(outIm)
    #ranges = np.max(outIm) - min
    #outIm = (outIm-min)/ranges

    #outIm[outIm < 0.3] = 0
    #outIm = outIm * 255
    #img=outIm*10000
    #
    #cv2.imshow('img',outIm)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return outIm


def FrangiFilter2DPytorchSmallParameter(I, FrangiBetaOne, FrangiBetaTwo):
    defaultoptions = {'FrangiScaleRange': (1, 5), 'FrangiScaleRatio':1,
                      'verbose': False, 'BlackWhite': True};

    options = defaultoptions
    sigmas = np.arange(options['FrangiScaleRange'][0], options['FrangiScaleRange'][1], options['FrangiScaleRatio'])
    sigmas.sort()

    beta = FrangiBetaOne
    c = FrangiBetaTwo
    #beta = 1
    #c = 450

    shape = (I.size()[1], I.size()[2], len(sigmas))
    ALLfiltered =[]
    #ALLangles = np.zeros(shape)

    # Frangi filter for all sigmas
    Rb = 0
    S2 = 0
    count = 0
    for i in range(len(sigmas)):
        # Show progress
        if (options['verbose']):
            print('Current Frangi Filter Sigma: ', sigmas[i])

        # Make 2D hessian
        Dxx, Dxy, Dyy = Hessian2DPytorch(I, sigmas[i], 3 * sigmas[i])
        #Dxx, Dxy, Dyy = Hessian2DPytorch(I, sigmas[i], 3)

        # Correct for scale
        Dxx = pow(sigmas[i], 2) * Dxx
        Dxy = pow(sigmas[i], 2) * Dxy
        Dyy = pow(sigmas[i], 2) * Dyy

        # Calculate (abs sorted) eigenvalues and vectors
        [Lambda2, Lambda1, Ix, Iy] = eig2imagePytorch(Dxx, Dxy, Dyy)

        # Compute the direction of the minor eigenvector
        angles = torch.atan2(Ix, Iy)

        # Compute some similarity measures
        Lambda1[Lambda1 == 0] = np.spacing(1)

        Rb = (Lambda2 / Lambda1) ** 2
        S2 = Lambda1 ** 2 + Lambda2 ** 2

        #a = torch.exp(-Rb / beta)
        #b = torch.ones(I.size()).double()
        # Compute the output image
        #BatchSize = beta.size()
        #beta = beta.view(beta.size(),1,1,1)
        Ifiltered = torch.exp(-Rb / beta) * (torch.ones(I.size()).double().cuda() - torch.exp(-S2 / c))

        # Enable the gradient to back probagation
        #Ifiltered[Lambda1 < 0] = 0
        Ifiltered = torch.nn.functional.relu(Lambda1) * Ifiltered

        # see pp. 45
        #if (options['BlackWhite']):
        #    Ifiltered[Lambda1 < 0] = 0
        #else:
        #    Ifiltered[Lambda1 > 0] = 0

        ALLfiltered.append(Ifiltered)
        # Return for every pixel the value of the scale(sigma) with the maximum
        # output pixel value

    if len(sigmas) > 1:
        ALLfiltered = torch.stack(ALLfiltered)
        outIm, outImIndex  = torch.max(ALLfiltered, dim=0)
    else:
        outIm = (Ifiltered.transpose()).reshape(I.shape)

    #outIm = torch.squeeze(outIm)
    #outIm = outIm.cpu().data.numpy()
    #min = np.min(outIm)
    #ranges = np.max(outIm) - min
    #outIm = (outIm-min)/ranges

    #outIm[outIm < 0.3] = 0
    #outIm = outIm * 255
    #img=outIm*10000
    #
    #cv2.imshow('img',outIm)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return outIm
# Read Images:
#imagename = "/home/yinpengshuai/Desktop/FrangiNet/Frangi-filter-based-Hessian-master/Screenshots/test.tif"
#image = cv2.imread(imagename, 0)
#blood = cv2.normalize(image.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
#Dxx1, Dxy1, Dyy1 = Hessian2D(blood, 1)
#Dxx, Dxy, Dyy = Hessian2DPytorch(blood, 1, 1)
#Lambda1, Lambda2, Ix, Iy = eig2imagePytorch(Dxx, Dxy, Dyy)
# outIm = FrangiFilter2DPytorch(blood)
# outIm = torch.squeeze(outIm)
# outIm = outIm.cpu().data.numpy()
# img=outIm*10000
#
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# a = 1
