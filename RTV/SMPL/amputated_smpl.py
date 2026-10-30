from scipy import sparse
import trimesh
import numpy as np
from PIL import Image
from io import BytesIO
from OffscreenRenderer.flat_renderer import FlatRenderer
from OffscreenRenderer.base_renderer import BaseRenderer
from OffscreenRenderer.uv_renderer import UVRenderer


class AmputatedSMPL:
    def __init__(self,):
        self.uv_map_matrix = sparse.load_npz("./assets/smpl/unwrap_sparse.npz")
        model = trimesh.load("assets/smpl/amputated_smpl_m_zero_uv.obj")
        self.f_uv = model.faces
        self.uv = model.visual.uv
        self.texPath = './assets/color_grid.png'#'./assets/smpl/amputated_smpl_m_zero_uv.png'
        self.flat_render = None#FlatRenderer(texPath=self.texPath)
        self.base_render = None#BaseRenderer()
        self.uv_render = None

    def get_amputated_vertex_uv_face(self,verts):
        amputated_verts = self.uv_map_matrix.dot(verts)
        return amputated_verts,self.uv,self.f_uv.reshape(-1)

    def render(self, verts, model, view, proj,height=512,width=512):
        if self.flat_render is None:
            #self.flat_render.__del__()
            self.flat_render = FlatRenderer(texPath=self.texPath,height=height,width=width)
        amputated_verts = self.uv_map_matrix.dot(verts)
        return self.flat_render.render(amputated_verts, self.uv,self.f_uv.reshape(-1),model,view,proj)

    def render_uv(self, verts, model, view, proj,height=512,width=512):
        if self.uv_render is None:
            self.uv_render = UVRenderer(texPath=self.texPath,height=height,width=width)
        amputated_verts = self.uv_map_matrix.dot(verts)
        return self.uv_render.render(amputated_verts, self.uv,self.f_uv.reshape(-1),model,view,proj)

    def render_base(self, verts, model, view, proj,height=512,width=512):
        if self.base_render is None:
            self.base_render = BaseRenderer(height=height,width=width)
        #print(verts.shape)
        amputated_verts = self.uv_map_matrix.dot(verts)
        #print(self.f_uv.reshape(-1).shape)
        return self.base_render.render(amputated_verts,self.f_uv.reshape(-1),model,view,proj)
