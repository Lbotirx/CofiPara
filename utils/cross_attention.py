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


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, embed_dim=768, k=256):
        super().__init__()
        self.img_tensor_shapes = []

        self.W_b = nn.Parameter(torch.randn(embed_dim, k))
        self.W_v = nn.Parameter(torch.randn(k, k))
        self.W_q = nn.Parameter(torch.randn(k, embed_dim))
        self.w_hv = nn.Parameter(torch.randn(k, 1))
        self.w_hq = nn.Parameter(torch.randn(k, 1))

        #self.W_w = nn.Parameter(torch.randn(embed_dim, embed_dim))
        #self.W_p = nn.Parameter(torch.randn(embed_dim*2, embed_dim))
        #self.W_s = nn.Parameter(torch.randn(embed_dim*2, embed_dim))

        self.W_w = nn.Linear(embed_dim, embed_dim)
        self.W_p = nn.Linear(embed_dim*2, embed_dim)
        self.W_s = nn.Linear(embed_dim*2, embed_dim)

        # self.fc = nn.Linear(embed_dim, num_classes)

    def flatten_img_tnsrs(self,srcs):
        """
        Flatten a input image tensor
        """
        src_flatten = []
        spatial_shapes = []
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

    def forward(self, text_tensor, image_tensor):
        # text_tensor: B x L x 512
        # image_tensor: B x 512 x 196
        # text_tensor: B x L x 768
        # image_tensor: B x \sum{HW} x 256

        v_word, q_word = self.parallel_co_attention(image_tensor, text_tensor)

        h_w = self.tanh(self.W_w(q_word + v_word))

        v_word = self.deflatten_img_tnsrs(v_word)

        return h_w, v_word

    def parallel_co_attention(self, V, Q):  # V : B x 512 x 196, Q : B x L x 512
        # V : B x 256 x hw, Q : B x L x 768
        C = torch.matmul(Q, torch.matmul(self.W_b, V)) # B x L x hw

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))    # B x k x hw
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1))) # B x k x L

        #a_v = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)) # B x 196
        #a_q = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)) # B x L

        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) # B x 1 x hw
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) # B x 1 x L

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # B x 256
        q = torch.squeeze(torch.matmul(a_q, Q))                  # B x 768
        # 维度有问题，再检查一下，或者直接跑一下看看

        return v, q
    
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
