import torch
import numpy as np
import trimesh
import time
import sys
import os

from pathlib import Path


class Editor:
    def __init__(self, runner):
        self.runner = runner

    def sample_foreground(
            self, 
            bending_latent,
            n_sample=64, 
            bbox=None, 
            rigidity_threshold=0.2,
            device='cpu',
            sdf_threshold=0.):
        if bbox is None:
            bbox = np.array([
                0, 0, 0,
                0, 0, 1,
                0, 1, 0,
                0, 1, 1,
                1, 0, 0,
                1, 0, 1,
                1, 1, 0,
                1, 1, 1
            ]) 
            bbox = (bbox - 0.5) * 3

        bbox = torch.from_numpy(np.array(bbox).reshape([-1, 3])).cuda()
        sample_pts = bbox[0][None, :].expand(n_sample ** 3, -1)
        idx = torch.arange(n_sample ** 3)[:, None].cuda()
        dirx, diry, dirz = (bbox[4] - bbox[0]), (bbox[2] - bbox[0]), (bbox[1] - bbox[0])  
        dirx = dirx[None, :].expand(n_sample ** 3, -1) / (n_sample - 1)
        diry = diry[None, :].expand(n_sample ** 3, -1) / (n_sample - 1)
        dirz = dirz[None, :].expand(n_sample ** 3, -1) / (n_sample - 1)

        sample_pts = sample_pts + torch.remainder(idx, n_sample) *  dirz \
            + torch.remainder(idx.div(n_sample, rounding_mode="floor" ), n_sample) * diry \
            + idx.div(n_sample, rounding_mode="floor" ).div(n_sample, rounding_mode="floor" ) * dirx
        sample_pts = sample_pts.float()

        all_indices = np.ones([(n_sample + 1) ** 3], dtype=int) * -1

        bending_details_0 = self.runner.sdf_network.bending_network[0](sample_pts, bending_latent, special_loss_return=True)
        sdf_value = self.runner.sdf_network.sdf(sample_pts, bending_latent)

        is_occupied = torch.logical_and(
            torch.le(sdf_value, sdf_threshold), 
            torch.gt(bending_details_0['rigidity_mask'], rigidity_threshold))

        dx = [0, 0, 0, 0, 1, 1, 1, 1]
        dy = [0, 0, 1, 1, 0, 0, 1, 1]
        dz = [0, 1, 0, 1, 0, 1, 0, 1]


        ''' 0412 5147 3217 6427 4217
          3.........7
        2 ..      ..
        y. .     . .
        ......... 6.
        .  .z   .  .
        .  .........
        . . 1   . .5
        ..      ..
        .........   x
        0       4
        ''' 

        ddd = [np.array([dx[i], dy[i], dz[i]]) - 0.5 for i in range(8)]
        sample_pts = sample_pts.detach().cpu().numpy()
        dirx, diry, dirz = dirx[0].cpu().numpy(), diry[0].cpu().numpy(), dirz[0].cpu().numpy()

        # is_debug = True
        # if is_debug:
        #     sample_pts = np.array([[0.5, 0.5, 0.5]])

        total_verts = 0
        eles = []
        verts = []


        def _idx(x, y, z, n):
            return int(x * n * n + y * n + z)

        extend = 1
        for i in range(n_sample ** 3):
            is_counted = False
            for ix in range(-extend, extend+1):
                for iy in range(-extend, extend+1):
                    for iz in range(-extend, extend+1):
                        idx = i + ix * n_sample * n_sample + iy * n_sample + iz
                        if idx >= 0 and idx < n_sample ** 3:
                            is_counted = is_counted or is_occupied[idx]
            if is_counted:
                ele = []
                x = i // n_sample // n_sample
                y = (i // n_sample) % n_sample
                z = i % n_sample
                for j in range(8):
                    idx = _idx(x + dx[j], y + dy[j], z + dz[j], n_sample + 1)
                    if all_indices[idx] == -1:
                        all_indices[idx] = total_verts
                        total_verts += 1
                        verts.append(sample_pts[i] + ddd[j][2] * dirz + ddd[j][1] * diry + ddd[j][0] * dirx)
                    ele.append(all_indices[idx])
                eles.append(ele)

        tets = []
        v2ts = [ [] for _ in range(total_verts)]
        num_tet = 0
        D = 24
        for i in range(n_sample ** 3):
            is_counted = False
            for ix in range(-1, 2):
                for iy in range(-1, 2):
                    for iz in range(-1, 2):
                        idx = i + ix * n_sample * n_sample + iy * n_sample + iz
                        if idx >= 0 and idx < n_sample ** 3:
                            is_counted = is_counted or is_occupied[idx]
            if is_counted:
                x = i // n_sample // n_sample
                y = (i // n_sample) % n_sample
                z = i % n_sample

                # 0412 5147 3217 6427 4217
                for tet_ids in [[0, 4, 1, 2], [5, 1, 4, 7], [3, 7, 2, 1], [4, 2, 7, 1], [6, 4, 2, 7]]:
                    tet = []
                    num_tet += 1
                    for j in tet_ids:
                        idx = _idx(x + dx[j], y + dy[j], z + dz[j], n_sample + 1)

                        v2ts[all_indices[idx]].append(len(tets))
                        tet.append(all_indices[idx])
                    tets.append(tet)

        tmp_a = []
        for v2t in v2ts:
            tmp_a.append(len(v2t))
            while len(v2t) < D:
                v2t.append(0)

        verts = np.array(verts)
        eles = np.array(eles, dtype=int)
        v2ts = np.array(v2ts, dtype=int)
        tets = np.array(tets, dtype=int)
        return verts, eles, tets, v2ts

    def compute_is_in_range(self, barycentric_coords, k, D):
        # n_q_pts x k x D
        is_in_range = torch.ones(
            [barycentric_coords.shape[0], k, D]).to(barycentric_coords.device)
        for i in range(4):
            is_in_range = torch.logical_and(
                is_in_range,
                torch.logical_and(
                    barycentric_coords[:, :, :, i] >= 0,
                    barycentric_coords[:, :, :, i] <= 1
                    )
                )

        is_in_range = is_in_range.reshape([is_in_range.shape[0], -1])
        vals, inds = is_in_range.max(dim=1)
        is_in_range = is_in_range.sum(1)
        is_in_range = torch.where(is_in_range > 0, 1, -1)
        return is_in_range, inds

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

    def get_bbox_ref(self, bbox):
        # verts, eles, tets, v2ts

        eles = [[0, 1, 2, 3, 4, 5, 6, 7]]
        tets = [[0, 4, 1, 2], [5, 1, 4, 7], [3, 2, 1, 7], [4, 2, 1, 7], [6, 4, 2, 7]]
        v2ts = [ [] for _ in range(8)]

        # 0412 5147 3217 6427 4217
        ii = 0
        for tet_ids in tets:
            for j in tet_ids:
                v2ts[j].append(ii)
            ii += 1

        D = 24
        for v2t in v2ts:
            while len(v2t) < D:
                v2t.append(0)

        verts = np.array(bbox).reshape([-1, 3]) 
        # verts = (verts - 0.5) * 3
        eles = np.array(eles, dtype=int)
        v2ts = np.array(v2ts, dtype=int)
        tets = np.array(tets, dtype=int)
     
        return verts, eles, tets, v2ts

    def move_fg_hand002(
        self, pts, latent,
        ):

        # bbox = [
        #     0, 0, 0,
        #     0, 0, 1,
        #     0, 1, 0,
        #     0, 1, 1,
        #     1, 0, 0,
        #     1, 0, 1,
        #     1, 1, 0,
        #     1, 1, 1
        # ]
        bbox = [
            -0.30467, -0.13148, -0.037142,
            -0.24081, -0.13148, 0.095085,
            -0.30467, -0.43884, -0.037142,
            -0.24081, -0.43884, 0.095085,
            0.19811, -0.13148, -0.37994,
            0.31532, -0.13148, -0.2072,
            0.19811, -0.43884, -0.37994,
            0.31532, -0.43884, -0.2072
        ]
        points_ref, eles_ref, tets, v2ts = self.get_bbox_ref(bbox)
        new_points_ref = points_ref.copy() 
        new_points_ref[:, 1] = points_ref[:, 1] + 0.06

        new_vertices = pts
        k = 8
        D = 18
        with torch.no_grad():
            points_ref = torch.from_numpy(
                points_ref).to(new_vertices.device).to(new_vertices.dtype)
            barycentric_coords, tets = self.compute_barycentric_coords(
                new_vertices, new_points_ref, eles_ref, tets, v2ts, 
                k=k, D=D, return_tets=True)

            # n_q_pts x k x D
            is_in_range, inds = self.compute_is_in_range(barycentric_coords, k, D)

            tets = tets.reshape([tets.shape[0], -1, 4])
            # n_q_pts x kD x 4
            barycentric_coords = barycentric_coords.reshape([
                barycentric_coords.shape[0], -1 ,4])

            back_verts = torch.zeros_like(
                new_vertices, dtype=new_vertices.dtype).to(new_vertices.device)
            
            qs_ref_points_id = torch.gather(
                tets,
                1,
                inds[:, None, None].expand(-1, 1, 4)
                ).squeeze(1)
            qs_bary_weights = torch.gather(
                barycentric_coords,
                1,
                inds[:, None, None].expand(-1, 1, 4)
                ).squeeze(1)

            for i in range(4):
                back_verts = back_verts + qs_bary_weights[:, i].unsqueeze(1) * points_ref[qs_ref_points_id[:, i]]
            back_verts[is_in_range < 0, :] = new_vertices[is_in_range < 0, :]
            diff = (back_verts - new_vertices).norm(dim=1)
            print("diff", diff.max(), diff.min(), diff.mean())
            return back_verts


    def duplicate_fg_hand002(
        self, pts, latent,
        ):

        bbox = [
            -0.30467, -0.13148, -0.037142,
            -0.24081, -0.13148, 0.095085,
            -0.30467, -0.43884, -0.037142,
            -0.24081, -0.43884, 0.095085,
            0.19811, -0.13148, -0.37994,
            0.31532, -0.13148, -0.2072,
            0.19811, -0.43884, -0.37994,
            0.31532, -0.43884, -0.2072
        ]
        points_ref, eles_ref, tets, v2ts = self.get_bbox_ref(bbox)
        new_points_ref = points_ref.copy() 
        new_points_ref = points_ref * 0.45

        new_vertices = pts
        k = 8
        D = 18
        with torch.no_grad():
            barycentric_coords, tets = self.compute_barycentric_coords(
                new_vertices, new_points_ref, eles_ref, tets, v2ts, 
                k=k, D=D, return_tets=True)

            # n_q_pts x k x D
            is_in_range, inds = self.compute_is_in_range(barycentric_coords, k, D)


            tets = tets.reshape([tets.shape[0], -1, 4])
            # n_q_pts x kD x 4
            barycentric_coords = barycentric_coords.reshape([
                barycentric_coords.shape[0], -1 ,4])

            back_verts = torch.zeros_like(
                new_vertices, dtype=new_vertices.dtype).to(new_vertices.device)
            
            qs_ref_points_id = torch.gather(
                tets,
                1,
                inds[:, None, None].expand(-1, 1, 4)
                ).squeeze(1)
            qs_bary_weights = torch.gather(
                barycentric_coords,
                1,
                inds[:, None, None].expand(-1, 1, 4)
                ).squeeze(1)

            points_ref = torch.from_numpy(
                points_ref).to(new_vertices.device).to(new_vertices.dtype)
            for i in range(4):
                back_verts = back_verts + qs_bary_weights[:, i].unsqueeze(1) * points_ref[qs_ref_points_id[:, i]]
            back_verts[is_in_range < 0, :] = new_vertices[is_in_range < 0, :]
            diff = (back_verts - new_vertices).norm(dim=1)
            return back_verts


    def color_fg_hand002(
        self, pts, latent,
        ):

        bbox = [
            -0.30467, -0.13148, -0.037142,
            -0.24081, -0.13148, 0.095085,
            -0.30467, -0.43884, -0.037142,
            -0.24081, -0.43884, 0.095085,
            0.19811, -0.13148, -0.37994,
            0.31532, -0.13148, -0.2072,
            0.19811, -0.43884, -0.37994,
            0.31532, -0.43884, -0.2072
        ]
        bbox = np.array(bbox).reshape([-1, 3])
        bbox = 1.2 * (bbox - bbox.mean(axis=0, keepdims=True)) + bbox.mean(axis=0, keepdims=True)


        n_sample = 8
        bbox, eles_ref, tets, v2ts = self.sample_foreground(
            latent, 
            n_sample=n_sample, 
            bbox=bbox,
            device='cuda',
            rigidity_threshold=0,
            sdf_threshold=2.
        )
        backed_ref = self.runner.color_network.bending_network[0](
            torch.tensor(bbox).float(), self.runner.renderer.bending_latents[0])

        k = 8
        D = 18

        sdf_nn_output = self.runner.sdf_network(pts, latent)
        sdf = sdf_nn_output[:, :1]
        if pts.requires_grad:
            feature_vector = sdf_nn_output[:, 1:]
            gradients = self.runner.sdf_network.gradient(pts, latent).squeeze()
            sampled_color = self.runner.color_network(
                pts, gradients, self.runner.renderer.tmp_dirs, feature_vector, latent)
            

            backed_pts = self.runner.color_network.bending_network[0](
                pts, latent)
            backed_ref = backed_ref.detach().cpu().numpy()

            barycentric_coords, tets = self.compute_barycentric_coords(
                backed_pts, backed_ref, eles_ref, tets, v2ts, 
                k=k, D=D, return_tets=True)

            is_in_range, inds = self.compute_is_in_range(barycentric_coords, k, D)


            sampled_color[is_in_range < 0, :] = \
                sampled_color[is_in_range < 0, :].mean(-1, keepdim=True)
            return sdf, sampled_color, gradients
        else:
            return sdf, None, None

    # edit_type == 'delete_fg'
    def delete_fg_hand002(
        self, pts, latent, 
        rigidity_threshold=0.2, is_delete_fg=True):

        bbox = [
            -0.30467, -0.13148, -0.037142,
            -0.24081, -0.13148, 0.095085,
            -0.30467, -0.43884, -0.037142,
            -0.24081, -0.43884, 0.095085,
            0.19811, -0.13148, -0.37994,
            0.31532, -0.13148, -0.2072,
            0.19811, -0.43884, -0.37994,
            0.31532, -0.43884, -0.2072
        ]
        bbox = np.array(bbox).reshape([-1, 3])
        bbox = 1.2 * (bbox - bbox.mean(axis=0, keepdims=True)) + bbox.mean(axis=0, keepdims=True)

        n_sample = 8
        bbox, eles_ref, tets, v2ts = self.sample_foreground(
            latent, 
            n_sample=n_sample, 
            bbox=bbox,
            device='cuda',
            rigidity_threshold=0,
            sdf_threshold=2.
        )
        backed_ref = self.runner.color_network.bending_network[0](
            torch.tensor(bbox).float(), self.runner.renderer.bending_latents[0])

        # points_ref, eles_ref, tets, v2ts = self.get_bbox_ref(bbox)
        # new_points_ref = points_ref.copy() 

        # new_vertices = pts
        k = 8
        D = 18
        with torch.no_grad():

            backed_pts = self.runner.color_network.bending_network[0](
                pts, latent)
            backed_ref = backed_ref.detach().cpu().numpy()

            barycentric_coords, tets = self.compute_barycentric_coords(
                backed_pts, backed_ref, eles_ref, tets, v2ts, 
                k=k, D=D, return_tets=True)

            is_in_range, inds = self.compute_is_in_range(barycentric_coords, k, D)

            sdf = self.runner.sdf_network.sdf(pts, latent)
            sdf[is_in_range > 0] = sdf[is_in_range > 0] * 0.0 + 1.0 
            
            return sdf


    def duplicate_fg_ball3(
        self, pts, latent,
        ):

        # frame 0
        bbox = [
            0.049, 0.166, -0.044,
            0.049, 0.166, 0.093,
            0.049, 0.039, -0.044,
            0.049, 0.039, 0.093,
            0.184, 0.166, -0.044,
            0.184, 0.166, 0.093,
            0.184, 0.039, -0.044,
            0.184, 0.039, 0.093
        ]
        points_ref, eles_ref, tets, v2ts = self.get_bbox_ref(bbox)
        new_points_ref = points_ref.copy() 
        new_points_ref[:, 1] = new_points_ref[:, 1] + 0.1
        new_points_ref = (new_points_ref - new_points_ref.mean(axis=0, keepdims=True)) * 0.5 \
            + new_points_ref.mean(axis=0, keepdims=True)

        new_vertices = pts
        k = 8
        D = 18
        with torch.no_grad():
            barycentric_coords, tets = self.compute_barycentric_coords(
                new_vertices, new_points_ref, eles_ref, tets, v2ts, 
                k=k, D=D, return_tets=True)

            # n_q_pts x k x D
            is_in_range, inds = self.compute_is_in_range(barycentric_coords, k, D)


            tets = tets.reshape([tets.shape[0], -1, 4])
            # n_q_pts x kD x 4
            barycentric_coords = barycentric_coords.reshape([
                barycentric_coords.shape[0], -1 ,4])

            back_verts = torch.zeros_like(
                new_vertices, dtype=new_vertices.dtype).to(new_vertices.device)
            
            qs_ref_points_id = torch.gather(
                tets,
                1,
                inds[:, None, None].expand(-1, 1, 4)
                ).squeeze(1)
            qs_bary_weights = torch.gather(
                barycentric_coords,
                1,
                inds[:, None, None].expand(-1, 1, 4)
                ).squeeze(1)

            points_ref = torch.from_numpy(
                points_ref).to(new_vertices.device).to(new_vertices.dtype)
            for i in range(4):
                back_verts = back_verts + qs_bary_weights[:, i].unsqueeze(1) * points_ref[qs_ref_points_id[:, i]]
            back_verts[is_in_range < 0, :] = new_vertices[is_in_range < 0, :]
            diff = (back_verts - new_vertices).norm(dim=1)
            return back_verts


    # edit_type == 'delete_fg'
    def delete_fg_ball3(
        self, pts, latent, 
        rigidity_threshold=0.2, is_delete_fg=True):

        # frame 0
        bbox = [
            0.049, 0.166, -0.044,
            0.049, 0.166, 0.093,
            0.049, 0.039, -0.044,
            0.049, 0.039, 0.093,
            0.184, 0.166, -0.044,
            0.184, 0.166, 0.093,
            0.184, 0.039, -0.044,
            0.184, 0.039, 0.093
        ]
        bbox = np.array(bbox).reshape([-1, 3])
        bbox = 1.2 * (bbox - bbox.mean(axis=0, keepdims=True)) + bbox.mean(axis=0, keepdims=True)

        n_sample = 8
        bbox, eles_ref, tets, v2ts = self.sample_foreground(
            latent, 
            n_sample=n_sample, 
            bbox=bbox,
            device='cuda',
            rigidity_threshold=0,
            sdf_threshold=2.
        )
        backed_ref = self.runner.color_network.bending_network[0](
            torch.tensor(bbox).float(), self.runner.renderer.bending_latents[0])

        k = 8
        D = 18
        with torch.no_grad():

            backed_pts = self.runner.color_network.bending_network[0](
                pts, latent)
            backed_ref = backed_ref.detach().cpu().numpy()

            barycentric_coords, tets = self.compute_barycentric_coords(
                backed_pts, backed_ref, eles_ref, tets, v2ts, 
                k=k, D=D, return_tets=True)

            is_in_range, inds = self.compute_is_in_range(barycentric_coords, k, D)

            sdf = self.runner.sdf_network.sdf(pts, latent)
            sdf[is_in_range > 0] = sdf[is_in_range > 0] * 0.0 + 1.0 
            
            return sdf

    # edit_type == 'delete_fg'
    def color_fg_ball3(
        self, pts, latent):
        # frame 0
        bbox = [
            0.049, 0.166, -0.044,
            0.049, 0.166, 0.093,
            0.049, 0.039, -0.044,
            0.049, 0.039, 0.093,
            0.184, 0.166, -0.044,
            0.184, 0.166, 0.093,
            0.184, 0.039, -0.044,
            0.184, 0.039, 0.093
        ]

        n_sample = 8
        bbox, eles_ref, tets, v2ts = self.sample_foreground(
            latent, 
            n_sample=n_sample, 
            bbox=bbox,
            device='cuda',
            rigidity_threshold=0,
            sdf_threshold=1.
        )
        backed_ref = self.runner.color_network.bending_network[0](
            torch.tensor(bbox).float(), self.runner.renderer.bending_latents[0])

        k = 8
        D = 18

        sdf_nn_output = self.runner.sdf_network(pts, latent)
        sdf = sdf_nn_output[:, :1]
        if pts.requires_grad:
            feature_vector = sdf_nn_output[:, 1:]
            gradients = self.runner.sdf_network.gradient(pts, latent).squeeze()
            sampled_color = self.runner.color_network(
                pts, gradients, self.runner.renderer.tmp_dirs, feature_vector, latent)
            

            backed_pts = self.runner.color_network.bending_network[0](
                pts, latent)
            backed_ref = backed_ref.detach().cpu().numpy()

            barycentric_coords, tets = self.compute_barycentric_coords(
                backed_pts, backed_ref, eles_ref, tets, v2ts, 
                k=k, D=D, return_tets=True)

            is_in_range, inds = self.compute_is_in_range(barycentric_coords, k, D)


            sampled_color[is_in_range < 0, :] = \
                sampled_color[is_in_range < 0, :].mean(-1, keepdim=True)
            return sdf, sampled_color, gradients
        else:
            return sdf, None, None


    def move_fg_ball3(
            self, pts, latent, points_after, points_ref, eles_ref, tets, v2ts
        ):

        new_points_ref = points_after

        new_vertices = pts
        k = 8
        D = 20

        barycentric_coords, tets_b = self.compute_barycentric_coords(
            new_vertices, new_points_ref, eles_ref, tets, v2ts, 
            k=k, D=D, return_tets=True)

        # n_q_pts x k x D
        is_in_range, inds = self.compute_is_in_range(barycentric_coords, k, D)

        barycentric_coords_ori, tets_ori = self.compute_barycentric_coords(
            new_vertices, points_ref, eles_ref, tets, v2ts, 
            k=k, D=D, return_tets=True)
        is_in_range_ori, inds_ori = self.compute_is_in_range(barycentric_coords_ori, k, D)


        tets_b = tets_b.reshape([tets_b.shape[0], -1, 4])
        # n_q_pts x kD x 4
        barycentric_coords = barycentric_coords.reshape([
            barycentric_coords.shape[0], -1 ,4])

        back_verts = torch.zeros_like(
            new_vertices, dtype=new_vertices.dtype).to(new_vertices.device)
        
        qs_ref_points_id = torch.gather(
            tets_b,
            1,
            inds[:, None, None].expand(-1, 1, 4)
            ).squeeze(1)
        qs_bary_weights = torch.gather(
            barycentric_coords,
            1,
            inds[:, None, None].expand(-1, 1, 4)
            ).squeeze(1)

        points_ref = torch.from_numpy(
            points_ref).to(new_vertices.device).to(new_vertices.dtype)
        for i in range(4):
            back_verts = back_verts + qs_bary_weights[:, i].unsqueeze(1) * points_ref[qs_ref_points_id[:, i]]
        back_verts[is_in_range < 0, :] = new_vertices[is_in_range < 0, :]
        diff = (back_verts - new_vertices).norm(dim=1)
        # print("diff", diff.max(), diff.min(), diff.mean())

        is_not_moved = torch.logical_and(is_in_range < 0, is_in_range_ori > 0)
        pts = back_verts
        sdf_nn_output = self.runner.sdf_network(pts, latent)
        sdf = sdf_nn_output[:, :1]
        sdf[is_not_moved] = sdf[is_not_moved] * 0 + 1

        if pts.requires_grad:
            feature_vector = sdf_nn_output[:, 1:]
            gradients = self.runner.sdf_network.gradient(pts, latent).squeeze()
            sampled_color = self.runner.color_network(
                pts, gradients, self.runner.renderer.tmp_dirs, feature_vector, latent)
            
            return sdf, sampled_color, gradients
        else:
            return sdf, None, None


    def run_diff_sim_warp(self, opt_var, points_ref, eles_ref, i_frame, 
        dt=1e-2, do_rendering=False):
        from models.warp_gravity import Ball3Gravity

        q = torch.from_numpy(points_ref).cuda()
        v = torch.zeros_like(q)

        q = q + torch.tensor([[0., 2.0, 0.0]]).cuda().float()
        q = q * 2.
        seed = 42

        env = Ball3Gravity(points_ref, eles_ref)
        np.random.seed(seed)

        sim_qs = []

        s_time = time.time()
        for frame in range(i_frame):
            v = v + opt_var
            # print("----", q.shape, q.reshape([-1, 3])[:,1].max())
            q, v = env.layer_step(q.float(), v.float(), env, frame)
            sim_qs.append(q / 2. - torch.tensor([[0., 2.0, 0.0]]).cuda().float())
        print("----", q.shape, q.reshape([-1, 3])[:,1].max())

        print("simulation time: ", time.time() - s_time)
        return sim_qs



    def compute_coupling_loss_warp(self, 
            input_var, points_ref, eles_ref, image_idx, n_frames):

        # points_ref = points_ref.clone().detach().cpu().numpy()
        # mesh = trimesh.Trimesh(points_ref, [])
        # mesh.export(os.path.join(self.runner.base_exp_dir, 'meshes', '{:0>8d}_timestep_{}_sample_voxel.ply'.format(
        #     self.runner.iter_step, image_idx)))

        losses = []
        
        sim_qs = self.run_diff_sim_warp(input_var, points_ref, eles_ref, n_frames, do_rendering=True)#.cuda()
        points_ref = torch.from_numpy(points_ref).float()
        
        backed_ref = self.runner.sdf_network.bending_network[0](
            points_ref.cuda(), self.runner.bending_latents_list[image_idx])

        # mesh = trimesh.Trimesh(backed_ref.clone().detach().cpu().numpy(), [])
        # mesh.export(os.path.join(self.runner.base_exp_dir, 'meshes', '{:0>8d}_timestep_{}_foreground_backed_ref.ply'.format(
        #     self.runner.iter_step, image_idx)))

        if self.physics_type in ['ball_gravity']:
            for idx in range(image_idx + 1, image_idx + n_frames):
                i = idx - image_idx - 1
                backed_after = self.runner.sdf_network.bending_network[0](
                    sim_qs[i].cuda(), self.runner.bending_latents_list[idx])
                loss = ((backed_after - backed_ref).norm(dim=-1)).mean()
                losses.append(loss.unsqueeze(0))

            # mesh = trimesh.Trimesh(backed_after.clone().detach().cpu().numpy(), [])
            # mesh.export(os.path.join(self.runner.base_exp_dir, 'meshes', '{:0>8d}_timestep_{}_foreground_backed_step{}.ply'.format(
                # self.runner.iter_step, idx, self.now_physics_i)))


        total_coupling_loss = torch.cat(losses).mean()
        return sim_qs, total_coupling_loss # * 10
