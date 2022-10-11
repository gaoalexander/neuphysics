import torch
import torch.nn.functional as F
import numpy as np
import mcubes


def extract_fields(bound_min, bound_max, resolution, query_func, N=64):
    print("Bound Min: {}\nBound Max: {}".format(bound_min, bound_max))
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
    print("len(X)", len(X))
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, N=64):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func, N=N)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 bending_latents,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.writer = None
        self.bending_latents = bending_latents
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)  # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, image_idx, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3), self.bending_latents[image_idx]).reshape(batch_size,
                                                                                                        n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    image_idx,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape
        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts, self.bending_latents[image_idx])
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts, self.bending_latents[image_idx]).squeeze()
        sampled_color = color_network(pts, gradients, dirs, feature_vector, self.bending_latents[image_idx]).reshape(
            batch_size, n_samples, 3)
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        alpha_out = alpha
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] + \
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        weights_out = weights[:, :n_samples]
        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:  # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'input_points': pts,
            'densities': alpha_out,
            'weights_out': weights_out
        }

    def render(self, rays_o, rays_d, near, far, image_idx, iter_i=None, perturb_overwrite=-1, background_rgb=None,
               cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples  # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3), self.bending_latents[image_idx]).reshape(batch_size,
                                                                                                        self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2 ** i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  image_idx,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    image_idx,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'input_points': ret_fine['input_points'],
            'densities': ret_fine['densities'],
            'weights_out': ret_fine['weights_out']
        }

    def extract_geometry(self, bound_min, bound_max, image_idx, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(
                                    pts,
                                    self.bending_latents[image_idx]))

    def extract_fore_back_ground(self, bound_min, bound_max, image_idx, resolution,
                                 threshold=0.0, rigidity_threshold=0.5, is_foreground=True):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: self.fore_back(
                                    pts, self.bending_latents[image_idx], is_foreground,
                                    rigidity_threshold=rigidity_threshold))

    def extract_voxel_mesh(self, points_ref, eles_ref, tets, v2ts,
                           bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: self.inside_voxel_mesh(
                                    pts, points_ref, eles_ref, tets, v2ts))

    def extract_editing(self, bound_min, bound_max, image_idx,
                        new_points_ref, points_ref, eles_ref, tets, v2ts, resolution, threshold=0.0):
        return extract_geometry(
            bound_min,
            bound_max,
            resolution=resolution,
            threshold=threshold,
            query_func=lambda pts: -self.sdf_network.sdf(
                self.move_back(pts, new_points_ref, points_ref, eles_ref, tets, v2ts),
                self.bending_latents[image_idx]),
            N=48
        )

    def compute_barycentric_coords(self, pts, points_ref, eles_ref, tets, v2ts, k=8, D=24, return_tets=False):
        tets = torch.from_numpy(tets).to(pts.device)
        v2ts = torch.from_numpy(v2ts).to(pts.device)
        points_ref = torch.from_numpy(points_ref).to(pts.device).to(pts.dtype)

        pts_binary = torch.ones([pts.shape[0]]).to(pts.device) * -1

        dis = pts[:, None, :].expand(-1, points_ref.shape[0], -1)
        # dis n_q_pts x n_ref
        dis = (dis - points_ref.unsqueeze(0)).norm(dim=2)
        dis_min = dis.min(1)[0]
        # inds: n_q_pts x k
        vals, inds = torch.topk(-dis, k, dim=1)

        # nearest_tet: n_q_pts x k x D
        nearest_tet = torch.gather(
            v2ts[:, None, :].expand(-1, k, -1),
            0,
            inds[:, :, None].expand(-1, -1, D)
        )

        # print("v2ts", v2ts)
        # print("inds", inds)
        # print("nearest_tet", nearest_tet)

        # exit()
        # tet_vert_id: n_q_pts x k x D x 4
        tet_vert_id = torch.gather(
            tets[:, None, None, :].expand(-1, k, D, -1),
            0,
            nearest_tet[:, :, :, None].expand(-1, -1, -1, 4)
        )

        # n_q_pts x k x D x 4 x 3
        tet_vert_coords = torch.gather(
            points_ref[:, None, None, None, :].expand(-1, k, D, 4, -1),
            0,
            tet_vert_id[:, :, :, :, None].expand(-1, -1, -1, -1, 3)
        )

        ones = torch.ones([pts.shape[0], k, D, 4, 1]).to(pts.device)
        det0 = torch.det(torch.cat([tet_vert_coords, ones], dim=-1))

        # n_q_pts x k x D x 3
        query_pts = pts[:, None, None, :].expand(-1, k, D, -1)

        # n_q_pts x k x D x 4
        barycentric_coords = torch.ones([pts.shape[0], k, D, 4]).to(pts.device)
        for i in range(4):
            det_i = tet_vert_coords.clone()
            det_i[:, :, :, i, :] = query_pts
            det_i = torch.det(torch.cat([det_i, ones], dim=-1)) / det0

            barycentric_coords[:, :, :, i] = det_i

        if return_tets:
            return barycentric_coords, tet_vert_id
        else:
            return barycentric_coords

    def delete_parts(self, vertices, points_ref, eles_ref, tets, v2ts, delete_bg=True):
        k = 8
        D = 24
        with torch.no_grad():
            # print("points_ref", points_ref)
            # print("vertices", vertices)
            pts = torch.from_numpy(vertices).cuda()
            barycentric_coords = self.compute_barycentric_coords(
                pts, points_ref, eles_ref, tets, v2ts,
                k=k, D=D, return_tets=False)
            # print(barycentric_coords)
            # print("================= barycentric_coords")
            # n_q_pts x k x D
            is_in_range = torch.ones(
                [barycentric_coords.shape[0], k, D]).to(barycentric_coords.device)
            # print("barycentric_coords")
            # print(barycentric_coords)
            for i in range(4):
                is_in_range = torch.logical_and(
                    is_in_range,
                    torch.logical_and(
                        barycentric_coords[:, :, :, i] >= 0,
                        barycentric_coords[:, :, :, i] <= 1
                    )
                )

            is_in_range = is_in_range.reshape([is_in_range.shape[0], -1])
            is_in_range = is_in_range.sum(1)
            is_in_range = torch.where(is_in_range > 0, 1, -1)
            # print(is_in_range.max())
            # print("is_in_range")
            # print(is_in_range)
            print("> and <", len(pts[is_in_range > 0]), len(pts[is_in_range < 0]))

            if delete_bg:
                pts = pts[is_in_range > 0]
            else:
                pts = pts[is_in_range < 0]
        # exit()
        return pts.cpu().numpy()

    def move_back(self, new_vertices, new_points_ref, points_ref, eles_ref, tets, v2ts):
        k = 8
        D = 18
        with torch.no_grad():
            points_ref = torch.from_numpy(
                points_ref).to(new_vertices.device).to(new_vertices.dtype)
            barycentric_coords, tets = self.compute_barycentric_coords(
                new_vertices, new_points_ref, eles_ref, tets, v2ts,
                k=k, D=D, return_tets=True)

            # n_q_pts x k x D
            is_in_range = torch.ones(
                [barycentric_coords.shape[0], k, D]).to(barycentric_coords.device)
            for i in range(4):
                is_in_range = torch.logical_and(
                    is_in_range,
                    torch.logical_and(
                        barycentric_coords[:, :, :, i] > 0,
                        barycentric_coords[:, :, :, i] < 1
                    )
                )

            is_in_range = is_in_range.reshape([is_in_range.shape[0], -1])
            vals, inds = is_in_range.max(dim=1)
            is_in_range = is_in_range.sum(1)
            is_in_range = torch.where(is_in_range > 0, 1, -1)

            tets = tets.reshape([tets.shape[0], -1, 4])
            # n_q_pts x kD x 4
            barycentric_coords = barycentric_coords.reshape([
                barycentric_coords.shape[0], -1, 4])

            back_verts = torch.zeros_like(
                new_vertices, dtype=new_vertices.dtype).to(new_vertices.device)

            qs_ref_points_id = torch.gather(
                tets,
                1,
                inds[:, None, None].expand(-1, 1, 4)
            ).squeeze(1)
            # print(barycentric_coords.device)
            # print(inds.device)
            qs_bary_weights = torch.gather(
                barycentric_coords,
                1,
                inds[:, None, None].expand(-1, 1, 4)
            ).squeeze(1)

            for i in range(4):
                back_verts = back_verts + qs_bary_weights[:, i].unsqueeze(1) * points_ref[qs_ref_points_id[:, i]]
            # print(new_vertices.dtype)
            # print(back_verts.dtype)
            # print("barycentric_coords.dtype", barycentric_coords.dtype)
            back_verts[is_in_range < 0, :] = new_vertices[is_in_range < 0, :]
            diff = (back_verts - new_vertices).norm(dim=1)
            print("diff", diff, diff.max(), diff.min())
            return back_verts

    def inside_voxel_mesh(self, pts, points_ref, eles_ref, tets, v2ts):
        # pts = pts.cpu()
        k = 8
        D = 24
        with torch.no_grad():
            barycentric_coords = self.compute_barycentric_coords(
                pts, points_ref, eles_ref, tets, v2ts, k=k, D=D)

            # n_q_pts x k x D
            is_in_range = torch.ones([pts.shape[0], k, D]).to(pts.device)
            for i in range(4):
                is_in_range = torch.logical_and(
                    is_in_range,
                    torch.logical_and(
                        barycentric_coords[:, :, :, i] > 0,
                        barycentric_coords[:, :, :, i] < 1
                    )
                )

            is_in_range = is_in_range.reshape([pts.shape[0], -1])
            is_in_range = torch.where(is_in_range.sum(1) > 0, 1., -1.)
            print("===is_in_range > 0", (is_in_range > 0).sum())
            print("===is_in_range < 0", (is_in_range < 0).sum())
            self.tmp_points = pts[is_in_range > 0]
            return is_in_range

    def fore_back(self, pts, latent, is_foreground, rigidity_threshold=0.2):
        bending_details_0 = self.sdf_network.bending_network[0](pts, latent, special_loss_return=True)
        rigidity = bending_details_0['rigidity_mask']

        sdf = self.sdf_network.sdf(pts, latent)
        if is_foreground:
            return torch.where(
                rigidity < rigidity_threshold,
                -1 * torch.ones_like(sdf), -sdf)
        else:
            return torch.where(
                rigidity > rigidity_threshold,
                -1 * torch.ones_like(sdf), -sdf)
