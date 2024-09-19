import torch
import torch.nn as nn
import torch.nn.functional as F



import models
from galerkin import simple_attn

import torch_dct as dct
import SCS_FE


def make_coord(shape, ranges=None, flatten=True):

    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret



class FCANO(nn.Module):

    def __init__(self, encoder_spec, width=256, blocks=16):
        super().__init__()
        self.width = width
        self.encoder = SCS_FE.make_SCS_FE(encoder_spec) #Use SCS_FE as the encoder
        self.conv00 = nn.Conv2d((24 + 2)*4+2, self.width, 1)

        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)
        
        self.conv_edsr_1 = nn.Conv2d(256,24,1)
        self.conv_edsr_2 = nn.Conv2d(256,24,1)

        self.fc_dct = nn.Conv2d(48, 24, 1)
        nn.init.zeros_(self.fc_dct.weight) 
        nn.init.zeros_(self.fc_dct.bias) 

        self.fc1_1 = nn.Conv2d(256, 24, 1)
        self.fc2 = nn.Conv2d(48, 1, 1)

        
        self.dct_edsr = dct_edsr()
        self.dct_cut_edsr_4 = dct_cut_edsr_4()

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat
        
    def query_rgb(self, coord, cell):      
        feat = (self.feat)
        grid = 0

        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:

                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)
                
        rel_cell = cell.clone()
        rel_cell[:,0] *= feat.shape[-2]
        rel_cell[:,1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)
         
        grid = torch.cat([*rel_coords, *feat_s, \
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,coord.shape[1],coord.shape[2])],dim=1)

        x = self.conv00(grid)

        #dct
        x_dct = x 
        x_24_1 = self.conv_edsr_1(x_dct)
        x_dct = self.dct_edsr(x_24_1)

        #cut dct
        x_24_2 = self.conv_edsr_2(x)
        dct_x = self.dct_cut_edsr_4(x_24_2)

        # dct cat
        dct = torch.cat([x_dct,dct_x],dim=1)
        dct = self.fc_dct(dct) #
 
        #spatial
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)
        x = self.fc1_1(x)
        x = torch.cat([x,dct],dim=1) 
   
        ret = self.fc2(F.gelu(x))
        

        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)




class no_edge(nn.Module): 
    def __init__(self):
        super(no_edge,self).__init__()
        self.width = 256
        self.blocks = 16
        self.conv0 = simple_attn(self.width, self.blocks)
        self.conv1 = simple_attn(self.width, self.blocks)


    def forward(self, x):
        b,c,w,h = x.shape
        x = dct.dct(x)
        x = x.cuda()
        w_mid = int(w/2)
        h_mid = int(h/2)
        x1 = x[...,0:w_mid,0:h_mid]

        x1 = dct.idct(x1)
        x1 = x1.cuda()
        x1 = self.conv0(x1,0)
        x1 = self.conv1(x1,1)
        x1 = dct.dct(x1)
        x1 = x1.cuda()

        x = torch.zeros(b,c,w,h)
        x = x.cuda()
        x[...,0:w_mid,0:h_mid] = x1
        x = x.cuda()

        x = dct.idct(x)
        x = x.cuda()

        return x

class dct_edsr(nn.Module):
    def __init__(self):
        super(dct_edsr,self).__init__()
        self.edsr = models.SCS_FE.make_edsr_body()
        pass

    def forward(self, x):
        x = dct.dct(x)
        x = x.cuda()
        x = self.edsr(x)
        x = dct.idct(x)   
        x = x.cuda()  
        return x



class dct_cut_edsr_4(nn.Module):
    def __init__(self):
        super(dct_cut_edsr_4,self).__init__()
        self.edsr_1 = models.SCS_FE.make_edsr_body()
        self.edsr_2 = models.SCS_FE.make_edsr_body()
        self.edsr_3 = models.SCS_FE.make_edsr_body()
        self.edsr_4 = models.SCS_FE.make_edsr_body()
        pass

    def forward(self, x):

        b,c,w,h = x.shape
        x = dct.dct(x)
        x = x.cuda()
        w_mid = int(w/2)
        h_mid = int(h/2)
        x1 = x[...,0:w_mid,0:h_mid]
        x2 = x[...,w_mid:w,0:h_mid]
        x3 = x[...,0:w_mid,h_mid:h]
        x4 = x[...,w_mid:w,h_mid:h]

        x1 = self.edsr_1(x1)
        x2 = self.edsr_2(x2)
        x3 = self.edsr_3(x3)
        x4 = self.edsr_4(x4)

        x[...,0:w_mid,0:h_mid] = x1
        x[...,w_mid:w,0:h_mid] = x2
        x[...,0:w_mid,h_mid:h] = x3
        x[...,w_mid:w,h_mid:h] = x4 
        x = dct.idct(x)   
        x = x.cuda()  
        return x