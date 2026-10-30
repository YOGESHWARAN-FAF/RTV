import argparse
import os

import platform

if platform.system() != "Windows":
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np

try:
    if platform.system() == "Windows":
        egl = None
    else:
        import OpenGL.EGL as egl
except Exception:
    egl = None
from OpenGL.GL import *
from OpenGL.GL import shaders
from PIL import Image

from OffscreenRenderer.gl_utils import create_opengl_context, init_frame_buffer

from OffscreenRenderer.off_screen_render import VertextAttribType, IndexType, generate_vao, set_shader_params, render, \
    read_texture

VertextAttribType = np.float32
IndexType = np.uint32
OpenglVertexAttrType = GL_FLOAT
OpenglTriangleIndexType = GL_UNSIGNED_INT


class FlatRenderer:
    def __init__(self, height=512, width=512, texPath="./assets/color_pattern/board_300x300.png", back_black=False, phong=False):
        self.display, self.egl_surf, self.opengl_context = create_opengl_context(
            (width, height))
        self.height = height
        self.width = width

        actual_tex = texPath if os.path.exists(texPath) else "./assets/color_pattern/board_300x300.png"
        self.texID = read_texture(actual_tex)
        self.vao = None
        self.vbo_pos = None
        self.vbo_uv = None
        self.ebo = None
        self.num_indices = 0
        self.pos_capacity = 0
        self.triangle_vertex_properties = None
        self.back_black = back_black
        if phong:
            from OffscreenRenderer.Shaders.backblack_phong_shaders import VERTEX_SHADER, GEOMETRY_SHADER, FRAGMENT_SHADER
            self.shader = shaders.compileProgram(
                shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                shaders.compileShader(GEOMETRY_SHADER, GL_GEOMETRY_SHADER),
                shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER), )
        else:
            if not back_black:
                from OffscreenRenderer.shaders import VERTEX_SHADER, FRAGMENT_SHADER
                self.shader = shaders.compileProgram(
                    shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                    shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER), )
            else:
                from OffscreenRenderer.Shaders.backblack_shaders import VERTEX_SHADER, GEOMETRY_SHADER, FRAGMENT_SHADER
                self.shader = shaders.compileProgram(
                    shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                    shaders.compileShader(GEOMETRY_SHADER, GL_GEOMETRY_SHADER),
                    shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER), )
        self.rendered_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.render_frame_object = init_frame_buffer(self.rendered_image)
        self._unbind_context()

    def _bind_context(self):
        if hasattr(self, 'display') and self.display is not None and egl is not None:
            try:
                egl.eglMakeCurrent(self.display, self.egl_surf, self.egl_surf, self.opengl_context)
                return
            except Exception:
                pass
        if hasattr(self, 'opengl_context') and self.opengl_context is not None:
            try:
                import glfw
                glfw.make_context_current(self.opengl_context)
                return
            except Exception:
                pass
            if hasattr(self.opengl_context, 'makeCurrent') and hasattr(self, 'egl_surf'):
                try:
                    self.opengl_context.makeCurrent(self.egl_surf)
                except Exception:
                    pass

    def _unbind_context(self):
        if hasattr(self, 'display') and self.display is not None and egl is not None:
            try:
                egl.eglMakeCurrent(self.display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT)
                return
            except Exception:
                pass
        if hasattr(self, 'opengl_context') and self.opengl_context is not None:
            try:
                import glfw
                glfw.make_context_current(None)
                return
            except Exception:
                pass
            if hasattr(self.opengl_context, 'doneCurrent'):
                try:
                    self.opengl_context.doneCurrent()
                except Exception:
                    pass

    def __del__(self):
        if hasattr(self, 'display') and self.display is not None and egl is not None:
            try:
                egl.eglDestroySurface(self.display, self.egl_surf)
                egl.eglDestroyContext(self.display, self.opengl_context)
            except Exception:
                pass
        elif hasattr(self, 'egl_surf') and self.egl_surf is not None:
            try:
                import glfw
                glfw.destroy_window(self.egl_surf)
            except Exception:
                pass

    def render(self, vertex_positions, vertex_texcoord, face_indices, matrix_model=None, matrix_view=None,
               matrix_proj=None):
        self._bind_context()
        try:
            if self.vao is None:
                self.generate_vao(vertex_positions, vertex_texcoord, face_indices)
            else:
                self.update_vao(vertex_positions, vertex_texcoord, face_indices)

            glBindFramebuffer(GL_FRAMEBUFFER, self.render_frame_object)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0, 0, 0.0, 1.0)
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            glDepthMask(GL_TRUE)
            glDisable(GL_CULL_FACE)
            glUseProgram(self.shader)

            set_shader_params(self.shader, matrix_model=matrix_model, matrix_view=matrix_view, matrix_proj=matrix_proj)

            glBindVertexArray(self.vao)
            render(GL_TRIANGLES, self.num_indices, self.texID)
            glBindVertexArray(0)

            glReadBuffer(GL_COLOR_ATTACHMENT0)
            data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            return np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
        finally:
            self._unbind_context()

    def generate_vao(self, vertex_positions: np.ndarray, texcoord: np.ndarray, face_indices: np.ndarray):
        pos = np.ascontiguousarray(vertex_positions, dtype=np.float32)
        uv = np.ascontiguousarray(texcoord, dtype=np.float32)
        indices = np.ascontiguousarray(face_indices, dtype=IndexType)
        self.num_indices = len(indices)
        self.pos_capacity = pos.nbytes

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # 1. Dynamic Position Buffer (VBO 0)
        self.vbo_pos = glGenBuffers(1)
        self.triangle_vertex_properties = self.vbo_pos
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, pos.nbytes, pos, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # 2. Static UV Buffer (VBO 1) - uploaded ONCE to GPU
        self.vbo_uv = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_uv)
        glBufferData(GL_ARRAY_BUFFER, uv.nbytes, uv, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # 3. Static Index Buffer (EBO) - uploaded ONCE to GPU
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def update_vao(self, vertex_positions: np.ndarray, texcoord: np.ndarray, face_indices: np.ndarray):
        pos = np.ascontiguousarray(vertex_positions, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        if pos.nbytes == self.pos_capacity:
            glBufferSubData(GL_ARRAY_BUFFER, 0, pos.nbytes, pos)
        else:
            glBufferData(GL_ARRAY_BUFFER, pos.nbytes, pos, GL_DYNAMIC_DRAW)
            self.pos_capacity = pos.nbytes
        glBindBuffer(GL_ARRAY_BUFFER, 0)

