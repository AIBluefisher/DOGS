# pylint: disable=E1101

import torch
import torch.nn as nn

from conerf.model.backbone.encodings import SinusoidalEncoder


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : i,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class VarNet(nn.Module):
    def __init__(
        self,
        posi_enc_dim: int,
        view_enc_dim: int,
        net_depth: int = 3,
        net_width: int = 64,
        use_position_offset: bool = False,
        num_gaussian_sets: int = 1,
        device: str = "cuda",
    ):
        super().__init__()

        if not use_position_offset:
            assert num_gaussian_sets == 1, "Only one set Gaussians should be used for defocus blur!"

        if use_position_offset:
            assert num_gaussian_sets > 1, "Must use multiple Gaussian sets for deblur motion!"

        self.device = device
        # If to predict position offsets for Gaussians.
        self.use_position_offset = use_position_offset
        # The number of set for motion blur Gaussians.
        self.num_gaussian_sets = num_gaussian_sets

        # self.posi_encoder = SinusoidalEncoder(3, 0, posi_enc_dim, True)
        # self.view_encoder = SinusoidalEncoder(3, 0, view_enc_dim, True)
        self.posi_encoder, embed_pos_channel = get_embedder(posi_enc_dim, 3)
        self.view_encoder, embed_view_channel = get_embedder(view_enc_dim, 3)

        # `7` is for scales and rotations.
        # input_channels = self.posi_encoder.latent_dim + self.view_encoder.latent_dim + 7
        input_channels = embed_pos_channel + embed_view_channel + 7

        hiddens = [nn.Linear(net_width, net_width) if i % 2 == 0 else nn.ReLU()
                   for i in range((net_depth - 1) * 2)]
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, net_width),
            nn.ReLU(),
            *hiddens,
        ).to(self.device)

        self.scale_head = nn.Linear(
            net_width, 3 * (num_gaussian_sets)).to(self.device)
        self.rotation_head = nn.Linear(
            net_width, 4 * (num_gaussian_sets)).to(self.device)
        if use_position_offset:  # Motion blur.
            self.position_head = nn.Linear(
                net_width, 3 * (num_gaussian_sets - 1)).to(self.device)
            self.position_head.apply(init_linear_weights)

        self.mlp.apply(init_linear_weights)
        self.scale_head.apply(init_linear_weights)
        self.rotation_head.apply(init_linear_weights)

    def forward(self, position, scales, rotations, view_dirs):
        pos_embed = self.posi_encoder(position)
        view_embed = self.view_encoder(view_dirs)

        feat = torch.cat([pos_embed, view_embed, scales, rotations], dim=-1)
        feat = self.mlp(feat)

        delta_scales = self.scale_head(feat)
        delta_rotations = self.rotation_head(feat)

        delta_positions = None
        if self.use_position_offset:
            delta_positions = self.position_head(feat)

        return delta_scales, delta_rotations, delta_positions
