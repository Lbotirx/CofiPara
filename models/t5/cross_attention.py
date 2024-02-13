import torch
import torch.nn as nn
import torch.nn.functional as fn

def flatten_img_tnsrs(srcs):
    """
    Flatten a input image tensor
    """
    src_flatten = []
    spatial_shapes = []
    for src in srcs:
        bs, c, h, w = src.shape
        spatial_shape = (h, w)
        spatial_shapes.append(spatial_shape)
        src = src.flatten(2).transpose(1, 2)                # bs, hw, c
        src_flatten.append(src)
        # mask_flatten.append(mask)
    src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
    return src_flatten

    
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, k_dim, v_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        # 定义线性投影层，用于将输入变换到多头注意力空间
        self.proj_q = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_k = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_v = nn.Linear(in_dim, v_dim * num_heads, bias=False)
		# 定义多头注意力的线性输出层
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, in_dim = x.size()
        # 对输入进行线性投影, 将每个头的查询、键、值进行切分和拼接
        q = self.proj_q(x).view(batch_size, seq_len, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k = self.proj_k(x).view(batch_size, seq_len, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v = self.proj_v(x).view(batch_size, seq_len, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        # 计算注意力权重和输出结果
        attn = torch.matmul(q, k) / self.k_dim**0.5   # 注意力得分
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)   # 注意力权重参数
        output = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)   # 输出结果
        # 对多头注意力输出进行线性变换和输出
        output = self.proj_o(output)
        
        return output

class CrossAttention(nn.Module):
    def __init__(self, in_dim1 = 256, in_dim2 = 768, num_heads = 1):
        super(CrossAttention, self).__init__()

        self.num_heads = num_heads
        self.k_dim = in_dim1
        self.v_dim = in_dim1

        self.img_tensor_shapes = []
        
        self.proj_q1 = nn.Linear(in_dim1, self.k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, self.k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, self.v_dim * num_heads, bias=False)

        self.proj_q2 = nn.Linear(in_dim2, self.k_dim * num_heads, bias=False)
        self.proj_k1 = nn.Linear(in_dim1, self.k_dim * num_heads, bias=False)
        self.proj_v1 = nn.Linear(in_dim1, self.v_dim * num_heads, bias=False)

        self.proj_o1 = nn.Linear(self.v_dim * num_heads, in_dim1)
        self.proj_o2 = nn.Linear(self.v_dim * num_heads, in_dim2)

    
    def flatten_img_tnsrs(self,srcs):
        """
        Flatten a input image tensor
        """
        src_flatten = []
        spatial_shapes = []
        self.img_tensor_shapes = []
        for src in srcs:
            self.img_tensor_shapes.append(src.shape)
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        return src_flatten 

    def deflatten_img_tnsrs(self,src_flatten):
        """
        restore srcs
        """
        srcs_restored = []
        srcs = torch.split(src_flatten, [shape[2]*shape[3] for shape in self.img_tensor_shapes], dim=1)
        for src,shape in zip(srcs,self.img_tensor_shapes):
            src_restored = src.transpose(1, 2).view(shape)
            srcs_restored.append(src_restored)
        return srcs_restored

        
    def forward(self, x1, x2, mask=None):
        # x1 is output of detr transformer output: bs, \sum{hxw}, c 
        batch_size, seq_len1, in_dim1 = x1.size()
        _, seq_len2, in_dim2 = x2.size()
        
        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        q2 = self.proj_q2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k1 = self.proj_k1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v1 = self.proj_v1(x1).view(batch_size, seq_len1, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        
        attn1 = torch.matmul(q1, k2) / self.k_dim**0.5
        attn2 = torch.matmul(q2, k1) / self.k_dim**0.5
        
        if mask is not None:
            attn1 = attn1.masked_fill(mask == 0, -1e9)
            attn2 = attn2.masked_fill(mask == 0, -1e9)
        
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)
        
        output1 = torch.matmul(attn1, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output2 = torch.matmul(attn2, v1).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len2, -1)
        output1 = self.proj_o1(output1)
        output2 = self.proj_o2(output2)

        # output1 = self.deflatten_img_tnsrs(output1)
        
        return output1,output2
