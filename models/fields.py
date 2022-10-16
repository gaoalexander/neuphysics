import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder


# This implementation is partially borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 bending_network=None,
                 bending_latent_size=32):
        super(SDFNetwork, self).__init__()

        self.bending_network = [bending_network]
        self.bending_latent_size = bending_latent_size
        self.embed_fn_fine = None
        self.skip_in = skip_in
        self.scale = scale

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, input_xyz, input_bending_latent):
        input_xyz = input_xyz * self.scale
        input_xyz = self.bending_network[0](input_xyz, input_bending_latent)

        if self.embed_fn_fine is not None:
            input_xyz = self.embed_fn_fine(input_xyz)

        x = input_xyz
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input_xyz], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x, bending_latent):
        return self.forward(x, bending_latent)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x, bending_latent):
        x.requires_grad_(True)
        y = self.sdf(x, bending_latent)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


# This implementation is partially borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True,
                 bending_network=None,
                 bending_latent_size=32):
        super().__init__()

        self.bending_network = [bending_network]
        self.bending_latent_size = bending_latent_size
        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0 and self.mode == 'idr':
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, input_xyz, normals, view_dirs, feature_vectors, input_bending_latent):
        input_xyz = self.bending_network[0](input_xyz, input_bending_latent)

        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([input_xyz, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([input_xyz, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([input_xyz, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        x = input_pts
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = F.relu(x)
            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(x)
            feature = self.feature_linear(x)
            x = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                x = self.views_linears[i](x)
                x = F.relu(x)

            rgb = self.rgb_linear(x)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)


class BendingNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 multires,
                 bending_latent_size,
                 rigidity_hidden_dimensions,
                 rigidity_network_depth):

        super(BendingNetwork, self).__init__()
        self.use_positionally_encoded_input = False
        self.input_ch = 3
        self.output_ch = 3
        self.bending_latent_size = bending_latent_size
        self.use_rigidity_network = True
        self.rigidity_hidden_dimensions = rigidity_hidden_dimensions
        self.rigidity_network_depth = rigidity_network_depth

        # simple scene editing. set to None during training.
        self.rigidity_test_time_cutoff = None
        self.test_time_scaling = None
        self.activation_function = F.relu  # F.relu, torch.sin
        self.hidden_dimensions = 64  # 32
        self.network_depth = 5  # 3 # at least 2: input -> hidden -> output
        self.skips = []  # do not include 0 and do not include depth-1
        use_last_layer_bias = False

        self.embed_fn_fine = None
        if multires > 0:
            embed_fn, self.input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            # dims[0] = input_ch

        self.network = nn.ModuleList(
            [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
            [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
             if i + 1 in self.skips
             else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
             for i in range(self.network_depth - 2)] +
            [nn.Linear(self.hidden_dimensions, self.output_ch, bias=use_last_layer_bias)])

        # initialize weights
        with torch.no_grad():
            for i, layer in enumerate(self.network[:-1]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)

            # initialize final layer to zero weights to start out with straight rays
            self.network[-1].weight.data *= 0.0
            if use_last_layer_bias:
                self.network[-1].bias.data *= 0.0

        if self.use_rigidity_network:
            self.rigidity_activation_function = F.relu  # F.relu, torch.sin
            self.rigidity_skips = []  # do not include 0 and do not include depth-1
            use_last_layer_bias = True
            self.rigidity_tanh = nn.Tanh()

            self.rigidity_network = nn.ModuleList(
                [nn.Linear(self.input_ch, self.rigidity_hidden_dimensions)] +
                [nn.Linear(self.input_ch + self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                 if i + 1 in self.rigidity_skips
                 else nn.Linear(self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                 for i in range(self.rigidity_network_depth - 2)] +
                [nn.Linear(self.rigidity_hidden_dimensions, 1, bias=use_last_layer_bias)])

            # initialize weights
            with torch.no_grad():
                for i, layer in enumerate(self.rigidity_network[:-1]):
                    if self.rigidity_activation_function.__name__ == "sin":
                        # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                        # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                        if type(layer) == nn.Linear:
                            a = (
                                1.0 / layer.in_features
                                if i == 0
                                else np.sqrt(6.0 / layer.in_features)
                            )
                            layer.weight.uniform_(-a, a)
                    elif self.rigidity_activation_function.__name__ == "relu":
                        torch.nn.init.kaiming_uniform_(
                            layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                        )
                        torch.nn.init.zeros_(layer.bias)

                # initialize final layer to zero weights
                self.rigidity_network[-1].weight.data *= 0.0
                if use_last_layer_bias:
                    self.rigidity_network[-1].bias.data *= 0.0

    def forward(self, input_pts, input_latent, details=None, special_loss_return=False):
        raw_input_pts = input_pts[:, :3]  # positional encoding includes raw 3D coordinates as first three entries

        if self.embed_fn_fine is not None:
            input_pts = self.embed_fn_fine(input_pts)

        if special_loss_return and details is None:
            details = {}

        input_latents = input_latent.expand(input_pts.size()[0], -1)
        x = torch.cat([input_pts, input_latents], -1)

        for i, layer in enumerate(self.network):
            x = layer(x)
            # SIREN
            if self.activation_function.__name__ == "sin" and i == 0:
                x *= 30.0
            if i != len(self.network) - 1:
                x = self.activation_function(x)
            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

        unmasked_offsets = x
        if details is not None:
            details["unmasked_offsets"] = unmasked_offsets

        if self.use_rigidity_network:
            x = input_pts
            for i, layer in enumerate(self.rigidity_network):
                x = layer(x)
                # SIREN
                if self.rigidity_activation_function.__name__ == "sin" and i == 0:
                    x *= 30.0
                if i != len(self.rigidity_network) - 1:
                    x = self.rigidity_activation_function(x)
                if i in self.rigidity_skips:
                    x = torch.cat([input_pts, x], -1)

            rigidity_mask = (self.rigidity_tanh(x) + 1) / 2  # close to 1 for nonrigid, close to 0 for rigid

            if self.rigidity_test_time_cutoff is not None:
                rigidity_mask[rigidity_mask <= self.rigidity_test_time_cutoff] = 0.0

        if self.use_rigidity_network:
            masked_offsets = rigidity_mask * unmasked_offsets
            if self.test_time_scaling is not None:
                masked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + masked_offsets  # skip connection
            if details is not None:
                details["rigidity_mask"] = rigidity_mask
                details["masked_offsets"] = masked_offsets
        else:
            if self.test_time_scaling is not None:
                unmasked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + unmasked_offsets  # skip connection

        if special_loss_return:  # used for compute_divergence_loss()
            return details
        else:
            return new_points
