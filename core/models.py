import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *
#from .blocks import M_Encoder
#from .blocks import M_Decoder
#from .blocks import M_Conv
import cv2
from .FrangiFilterPytorch import FrangiFilter2DPytorchSmall, FrangiFilter2DPytorchSmallParameter

class FM_Net_SelfLearning(nn.Module):
    def __init__(self, n_classes, bn=True, BatchNorm=False):
        super(FM_Net_SelfLearning, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)
        self.flattens = nn.Flatten()
        self.linear = nn.Linear(512 * 32 * 32, 512)
        self.linearParameter = nn.Linear(512, 2)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x, xgrey):
        _, _, img_shape, _ = x.size()

        # These Frangi Filters serve as attentions.
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')

        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        out1 = self.flattens(out)
        para1 = self.linear(out1)
        para2 = self.linearParameter(para1)
        para2 = torch.sigmoid(para2).double() + 1

        FilterdResponse = FrangiFilter2DPytorchSmallParameter(xgrey, para2[:,0], para2[:,1])
        # Normalize to [0,1]
        min = torch.min(FilterdResponse)
        range = torch.max(FilterdResponse) - min
        NormalizedFilterResponse = (FilterdResponse - min) / range

        x_2_grey = F.upsample(xgrey, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        FilterdResponse_x_2 = FrangiFilter2DPytorchSmallParameter(x_2_grey, para2[:,0], para2[:,1])
        min = torch.min(FilterdResponse_x_2)
        range = torch.max(FilterdResponse_x_2) - min
        NormalizedFilterResponse_2 = (FilterdResponse_x_2 - min) / range

        x_3_grey = F.upsample(xgrey, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        FilterdResponse_x_3 = FrangiFilter2DPytorchSmallParameter(x_3_grey, para2[:,0], para2[:,1])
        min = torch.min(FilterdResponse_x_3)
        range = torch.max(FilterdResponse_x_3) - min
        NormalizedFilterResponse_3 = (FilterdResponse_x_3 - min) / range

        x_4_grey = F.upsample(xgrey, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        FilterdResponse_x_4 = FrangiFilter2DPytorchSmallParameter(x_4_grey, para2[:,0], para2[:,1])
        min = torch.min(FilterdResponse_x_4)
        range = torch.max(FilterdResponse_x_4) - min
        NormalizedFilterResponse_4 = (FilterdResponse_x_4 - min) / range

        up5 = self.up5(conv4, out)
        up5 = up5 + NormalizedFilterResponse_4.float()

        up6 = self.up6(conv3, up5)
        up6 = up6 + NormalizedFilterResponse_3.float()

        up7 = self.up7(conv2, up6)
        up7 = up7 + NormalizedFilterResponse_2.float()

        up8 = self.up8(conv1, up7)

        NormalizedFilterResponse2 = NormalizedFilterResponse[0,0,:,:].cpu().data.numpy()
        #cv2.imwrite('NormalizedFilterResponse.jpg', NormalizedFilterResponse2 * 256)

        up8 = up8 + NormalizedFilterResponse.float()

        FeatureMap2 = up8[0, 2, :, :].cpu().data.numpy()
        #cv2.imwrite('FeatureMap2.jpg', FeatureMap2 * 256)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
         #return [ave_out, side_5, side_6, side_7, side_8, FeatureMap, NormalizedFilterResponse2, FeatureMap2]
        return [ave_out, side_5, side_6, side_7, side_8]