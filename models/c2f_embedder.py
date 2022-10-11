import math
import torch


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class C2FEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x, iter: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, iter_idx, idx, p_fn=p_fn, freq=freq: self.c2f_weight(self.alpha(iter_idx), idx) * p_fn(
                        x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def alpha(self, iter_idx):
        return (iter_idx * self.kwargs['num_freqs']) / self.kwargs['c2f_anneal']

    def c2f_weight(self, alpha, freq_idx):
        def clamp(num, min_value, max_value):
            return max(min(num, max_value), min_value)

        return (1 - math.cos(math.pi * clamp(alpha - freq_idx, 0, 1))) / 2

    def embed(self, inputs, iter_idx):
        for idx, fn in enumerate(self.embed_fns):
            if idx == 0:
                embedding = fn(inputs, iter_idx)
            else:
                embedding = torch.cat([embedding, fn(inputs, iter_idx, (idx - 1) // 2)], -1)
        return embedding


def get_c2f_embedder(multires, c2f_anneal=100000, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
        'c2f_anneal': c2f_anneal
    }

    embedder_obj = C2FEmbedder(**embed_kwargs)

    def embed(x, iter_idx, eo=embedder_obj):
        return eo.embed(x, iter_idx)

    return embed, embedder_obj.out_dim
