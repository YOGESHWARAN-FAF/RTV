from bev import BEV
from bev.post_parser import denormalize_cam_params_to_trans, suppressing_redundant_prediction_via_projection, remove_outlier, body_mesh_projection2image

class MyBEV(BEV):
    def __init__(self, settings, fix_body=False):
        super(MyBEV, self).__init__(settings)
        self.fix_body = fix_body

    def forward(self, image, signal_ID=0, **kwargs):

        outputs = self.process_normal_image(image, signal_ID)
        if outputs is None:
            return None

        #if self.settings.render_mesh:
        #    mesh_color_type = 'identity' if self.settings.mode!='webcam' and not self.settings.save_video else 'same'
        #    rendering_cfgs = {'mesh_color':mesh_color_type, 'items': self.visualize_items, 'renderer': self.settings.renderer}
        #    outputs = rendering_romp_bev_results(self.renderer, outputs, image, rendering_cfgs)

        return outputs

    def process_normal_image(self, image, signal_ID):
        outputs, image_pad_info = self.single_image_forward(image)
        meta_data = {'input2org_offsets': image_pad_info}

        if outputs is None:
            return None

        if self.settings.temporal_optimize:
            outputs = self.temporal_optimization(outputs, signal_ID)
            if outputs is None:
                return None
            outputs.update({'cam_trans': denormalize_cam_params_to_trans(outputs['cam'])})

        #print(outputs['smpl_betas'].dtype, outputs['smpl_betas'].shape)#torch.Size([1, 11]
        if self.fix_body:
            outputs['smpl_betas'][:,10]=0
            outputs['smpl_betas'][:, 0:4] = 0
        if self.settings.calc_smpl:
            verts, joints, face = self.smpl_parser(outputs['smpl_betas'], outputs['smpl_thetas'])
            outputs.update({'verts': verts, 'joints': joints, 'smpl_face': face})
            if self.settings.render_mesh:
                meta_data['vertices'] = outputs['verts']
            projection = body_mesh_projection2image(outputs['joints'], outputs['cam'], **meta_data)
            outputs.update(projection)

            outputs = suppressing_redundant_prediction_via_projection(outputs, image.shape,
                                                                      thresh=self.settings.nms_thresh)
            outputs = remove_outlier(outputs, relative_scale_thresh=self.settings.relative_scale_thresh)
        return outputs


