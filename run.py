import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, BendingNetwork
from models.renderer import NeuSRenderer

from models.editor import Editor

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.divergence_loss_weight = self.conf.get_float('train.divergence_loss_weight')
        self.offsets_loss_weight = self.conf.get_float('train.offsets_loss_weight')
        self.rigidity_loss_weight = self.conf.get_float('train.rigidity_loss_weight')
        self.bending_latent_size = self.conf.get_int('train.bending_latent_size')

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        self.bending_latents_list = [torch.zeros(self.bending_latent_size).to(self.device)
                                     for i in range(self.dataset.n_images)]
        for each in self.bending_latents_list:
            each.requires_grad = True

        # Networks
        params_to_train = []

        self.bending_network = BendingNetwork(**self.conf['model.bending_network']).to(self.device)
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], bending_network=self.bending_network).to(
            self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network'],
                                              bending_network=self.bending_network).to(self.device)

        params_to_train += self.bending_latents_list
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        params_to_train += list(self.bending_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.bending_latents_list,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        self.renderer.bending_latents = self.bending_latents_list

        self.editor = Editor(self)
        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    # from FFJORD github code
    def divergence_approx(self, input_points, offsets_of_inputs):
        # avoids explicitly computing the Jacobian
        e = torch.randn_like(offsets_of_inputs, device=offsets_of_inputs.get_device())
        e_dydx = torch.autograd.grad(offsets_of_inputs, input_points, e, create_graph=True)[
            0
        ]
        e_dydx_e = e_dydx * e
        approx_tr_dydx = e_dydx_e.view(offsets_of_inputs.shape[0], -1).sum(dim=1)
        return approx_tr_dydx

    def compute_divergence_loss(self, details, input_points, densities, backprop_into_weights=False):
        weights = 1.0 - torch.exp(-F.relu(densities))
        offsets = (details["masked_offsets"]
                   if "masked_offsets" in details
                   else details["unmasked_offsets"])
        divergence_approx = self.divergence_approx(input_points, offsets)
        divergence_loss = torch.abs(divergence_approx) ** 2

        if weights is not None:
            if not backprop_into_weights:
                weights = weights.detach()
            divergence_loss = weights * divergence_loss
        return torch.mean(divergence_loss.view(self.batch_size, -1), dim=-1)

    def compute_offsets_loss(self, details, weights):
        rigidity_mask = details["rigidity_mask"].view(-1)
        offsets = details["unmasked_offsets"].view(-1, 3)

        offsets_loss = torch.mean((weights * torch.linalg.norm(offsets, ord=1, dim=-1)).view(self.batch_size, -1),
                                  dim=-1)
        offsets_loss += self.rigidity_loss_weight * torch.mean((weights * rigidity_mask).view(self.batch_size, -1),
                                                               dim=-1)
        return offsets_loss

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.renderer.writer = self.writer

        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            for each in self.bending_latents_list:
                each.grad = None

            image_idx = image_perm[self.iter_step % len(image_perm)]
            data = self.dataset.gen_random_rays_at(image_idx, self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far, image_idx,
                                              iter_i=iter_i,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            input_points = torch.clone(render_out['input_points'].view(-1, 3)).detach()
            input_points.requires_grad = True

            weights = torch.reshape(render_out['weights_out'].detach(), (-1,))
            densities = render_out['densities'].view(-1)

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            bending_net_details = self.bending_network(input_points,
                                                       self.bending_latents_list[image_idx],
                                                       special_loss_return=True)
            offsets_loss = self.compute_offsets_loss(bending_net_details, weights)
            divergence_loss = self.compute_divergence_loss(bending_net_details,
                                                           input_points,
                                                           densities,
                                                           backprop_into_weights=False)

            offsets_loss = torch.mean(offsets_loss)
            divergence_loss = torch.mean(divergence_loss)

            loss_term_schedule_weight = (1.0 / 100.0) ** (1 - iter_i / res_step)

            # Total Weighted Loss Function
            loss = color_fine_loss + \
                   eikonal_loss * self.igr_weight + \
                   mask_loss * self.mask_weight + \
                   divergence_loss * self.divergence_loss_weight * loss_term_schedule_weight + \
                   offsets_loss * self.offsets_loss_weight * loss_term_schedule_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(image_idx=0)

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)

        self.bending_network.load_state_dict(checkpoint['bending_network'])
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        self.bending_latents_list = checkpoint['bending_latents_list']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'bending_network': self.bending_network.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            'bending_latents_list': self.bending_latents_list
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              image_idx=idx,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])


    def train_physics_gravity_warp(self):
        idx = 0
        n_frames = 24
        rigidity_threshold = 0.2
        sdf_threshold = 0.01
        device = 'cuda:0'
        n_sample = 5
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
        # # frame 23
        bbox23 = [
            0.049, 0.27151, -0.044,
            0.049, 0.27151, 0.093,
            0.049, 0.14486, -0.044,
            0.049, 0.14486, 0.093,
            0.184, 0.27151, -0.044,
            0.184, 0.27151, 0.093,
            0.184, 0.14486, -0.044,
            0.184, 0.14486, 0.093
        ]
        points_ref, eles_ref, tets_ref, v2ts = self.editor.sample_foreground(
            self.renderer.bending_latents[idx], 
            n_sample=n_sample, 
            bbox=bbox,
            device=device,
            rigidity_threshold=rigidity_threshold,
            sdf_threshold=sdf_threshold
        )

        # os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        # mesh = trimesh.Trimesh(points_ref, [])
        # mesh.export(os.path.join(self.base_exp_dir, 'meshes', 'points_ref.ply'))

        points_23, eles_23, tets, v2ts = self.editor.sample_foreground(
            self.renderer.bending_latents[idx], 
            n_sample=n_sample, 
            bbox=bbox23,
            device=device,
            rigidity_threshold=rigidity_threshold,
            sdf_threshold=sdf_threshold
        )
        points_23 = torch.tensor(points_23).cuda().float()
        
        self.editor.physics_type = 'ball_gravity' # ball_gravity

        if self.editor.physics_type == 'ball_gravity':
            opt_var = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
            self.physics_optimizer = torch.optim.Adam(
                [opt_var], lr=0.1)

        physics_epoch = 20
        for i in range(physics_epoch):

            input_var = torch.cat([
                torch.tensor([0]),
                opt_var,
                torch.tensor([0]),
                ])
            # print(input_var.shape)

            self.editor.now_physics_i = i
            sim_qs, coupling_loss = self.editor.compute_coupling_loss_warp(
                input_var, points_ref, tets_ref, idx, n_frames)
            
            print("======================== epoch {:04d}".format(i))
            print("optimized and target", sim_qs[-1].mean(0), points_23.mean(0))
            self.physics_optimizer.zero_grad()
            coupling_loss.backward()
            print("opt_var", opt_var)

            self.physics_optimizer.step()


    def render_novel_image(self, idx_0, idx_1, ratio, timestep, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              image_idx=timestep,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def render_original_motion(self, idx_0, idx_1, ratio, timestep, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              image_idx=timestep,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def fore_back_ground(self, image_idx, world_space=False, resolution=64, threshold=0.0, rigidity_threshold=0.5):
        """
        Export a mesh with foreground separated from background, based on rigidity mask threshold and scene bounding box
        """
        print("fore_back_ground")

        pcd = trimesh.load(os.path.join(self.dataset.data_dir, 'sparse_points_interest.ply'))
        vertices = pcd.vertices
        bound_min = np.min(vertices, axis=0)
        bound_max = np.max(vertices, axis=0)
        center = (bound_min + bound_max) / 2
        radius = abs(np.linalg.norm(bound_max - center, ord=2))

        shrink_border_percent = 0.1

        bound_min = torch.tensor((bound_min - center) / radius * (1 - shrink_border_percent), dtype=torch.float32)
        bound_max = torch.tensor((bound_max - center) / radius * (1 - shrink_border_percent), dtype=torch.float32)

        # bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        # bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_fore_back_ground(bound_min, bound_max, image_idx,
                                                   resolution=resolution,
                                                   threshold=threshold,
                                                   rigidity_threshold=rigidity_threshold,
                                                   is_foreground=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes',
                                 '{:0>8d}_timestep_{}_foreground.ply'.format(self.iter_step, image_idx)))

        logging.info('End')

    def validate_mesh(self, image_idx, world_space=False, resolution=64, threshold=0.0, rigidity_test_time_cutoff=None):
        self.bending_network.rigidity_test_time_cutoff = rigidity_test_time_cutoff

        pcd = trimesh.load(os.path.join(self.dataset.data_dir, 'sparse_points_interest.ply'))
        vertices = pcd.vertices
        bound_min = np.min(vertices, axis=0)
        bound_max = np.max(vertices, axis=0)
        center = (bound_min + bound_max) / 2
        radius = abs(np.linalg.norm(bound_max - center, ord=2))

        shrink_border_percent = 0.1

        bound_min = torch.tensor((bound_min - center) / radius * (1 - shrink_border_percent), dtype=torch.float32)
        bound_max = torch.tensor((bound_max - center) / radius * (1 - shrink_border_percent), dtype=torch.float32)

        # bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        # bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, image_idx, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(
            os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_timestep_{}.ply'.format(self.iter_step, image_idx)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1, n_frames=60, rigidity_test_time_cutoff=None):
        self.bending_network.rigidity_test_time_cutoff = rigidity_test_time_cutoff

        images = []
        n_frames = n_frames
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)

        for i in range(n_frames):
            print(i)
            image = self.render_novel_image(img_idx_0,
                                            img_idx_1,
                                            np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                                            i,
                                            resolution_level=4)
            images.append(image)
            output_path = os.path.join(video_dir, '{:0>8d}_timestep_{}.png'.format(self.iter_step, i))
            cv.imwrite(output_path, image)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))
        for image in images:
            writer.write(image)

        writer.release()

    def original_motion(self, img_idx_0, img_idx_1, n_frames=1, timestep=0, rigidity_test_time_cutoff=None):
        self.bending_network.rigidity_test_time_cutoff = rigidity_test_time_cutoff

        images = []
        n_frames = n_frames
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)

        image = self.render_original_motion(img_idx_0,
                                            img_idx_1,
                                            np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                                            timestep,
                                            resolution_level=4)
        images.append(image)
        output_path = os.path.join(video_dir, '{:0>8d}_timestep_{}.png'.format(self.iter_step, i))
        cv.imwrite(output_path, image)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--val_frame_idx', type=int, default=0)
    parser.add_argument('--val_rigidity_threshold', type=float, default=0.0)
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--seq', type=str, default='ball3') # ball3 or hand002
    parser.add_argument('--edit_type', type=str, default='delete_fg') # move_fg delete_fg duplicate_fg
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()

    elif args.mode == 'validate_mesh':
        runner.validate_mesh(image_idx=args.val_frame_idx,
                             world_space=True,
                             resolution=512,
                             threshold=args.mcube_threshold,
                             rigidity_test_time_cutoff=args.val_rigidity_threshold)

    elif args.mode == 'validate_image':
        runner.validate_image(idx=args.val_frame_idx)

    elif args.mode == 'fore_back_ground':
        runner.fore_back_ground(image_idx=args.val_frame_idx,
                                world_space=True,
                                resolution=512,
                                threshold=args.mcube_threshold,
                                rigidity_threshold=args.val_rigidity_threshold)

    elif args.mode == 'validate_mesh_sequence':
        n_frames = runner.dataset.n_images
        for i in range(n_frames):
            print("Validating mesh {}/{}...".format(i, n_frames - 1))
            runner.fore_back_ground(image_idx=i,
                                    world_space=True,
                                    resolution=512,
                                    threshold=args.mcube_threshold,
                                    rigidity_threshold=args.val_rigidity_threshold)

    elif args.mode == 'validate_image_sequence':
        n_frames = runner.dataset.n_images
        for i in range(n_frames):
            print("Validating image {}/{}...".format(i, n_frames - 1))
            runner.interpolate_view(i, i, n_frames=1)

    elif args.mode == 'validate_image_sequence_original_motion':
        n_frames = runner.dataset.n_images
        for i in range(n_frames):
            print("Validating image {}/{}...".format(i, n_frames - 1))
            runner.original_motion(i, i, n_frames=1, timestep=i)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices

        n_frames = runner.dataset.n_images
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0,
                                img_idx_1,
                                n_frames=n_frames,
                                rigidity_test_time_cutoff=args.val_rigidity_threshold)

    elif args.mode == 'train_physics_gravity_warp':
        runner.train_physics_gravity_warp()
