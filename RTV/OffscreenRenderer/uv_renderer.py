import argparse
import os
import cv2

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
from OffscreenRenderer.uv_shaders import VERTEX_SHADER, FRAGMENT_SHADER
from OffscreenRenderer.off_screen_render import VertextAttribType, IndexType, generate_vao, set_shader_params, render, \
    read_texture

VertextAttribType = np.float32
IndexType = np.uint32
OpenglVertexAttrType = GL_FLOAT
OpenglTriangleIndexType = GL_UNSIGNED_INT


class UVRenderer:
    def __init__(self, height=512, width=512, texPath="./assets/measurement.png",black_bg=True):
        self.display, self.egl_surf, self.opengl_context = create_opengl_context(
            (width, height))
        self.height = height
        self.width = width
        self.black_bg = black_bg

        self.texID = read_texture(texPath) if texPath is not None else None
        self.vao = None
        self.triangle_vertex_properties = None
        self.shader = shaders.compileProgram(
            shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
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

    def update_texture(self,texPath):
        self._bind_context()
        try:
            if self.texID is not None:
                glDeleteTextures(self.texID)
            self.texID = read_texture(texPath)
        finally:
            self._unbind_context()

    def __del__(self):
        #glDeleteFramebuffers(1, self.render_frame_object)
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
            if self.black_bg:
                glClearColor(0, 0, 0.0, 1.0)
            else:
                glClearColor(1,1,1,1)
            # glEnable(GL_MULTISAMPLE)
            # glDisable(GL_MULTISAMPLE)
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            glDepthMask(GL_TRUE)

            glEnable(GL_CULL_FACE)
            glCullFace(GL_FRONT)
            glUseProgram(self.shader)
            # Orto projection in range [0; 1]
            set_shader_params(self.shader, matrix_model=matrix_model, matrix_view=matrix_view, matrix_proj=matrix_proj)

            glBindVertexArray(self.vao)
            render(GL_TRIANGLES, len(face_indices), self.texID)
            glBindVertexArray(0)

            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glBindFramebuffer(GL_FRAMEBUFFER, self.render_frame_object)

            glReadBuffer(GL_COLOR_ATTACHMENT0)

            data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)

            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            # glDeleteFramebuffers(1, render_frame_object)
            glUseProgram(self.shader)

            image = Image.frombuffer("RGB", (self.width, self.height), data)
            image = np.array(image)
            return image[:, :, 0:2]
        finally:
            self._unbind_context()

    def generate_vao(self, vertex_positions: np.ndarray, texcoord: np.ndarray, face_indices: np.ndarray):
        assert vertex_positions.ndim == 2
        assert vertex_positions.shape[1] == 3
        assert vertex_positions.shape[0] == texcoord.shape[0]
        assert texcoord.shape[1] == 2
        assert face_indices.ndim == 1

        face_indices = face_indices.astype(IndexType)

        vertex_attributes = np.hstack(
            (vertex_positions, texcoord)).astype(VertextAttribType)

        num_pos_per_vertex = vertex_positions.shape[1]
        num_texcoord_per_vertex = texcoord.shape[1]

        triangle_vao = glGenVertexArrays(1)
        self.vao = triangle_vao

        glBindVertexArray(triangle_vao)

        num_properties_per_vertex = vertex_attributes.shape[1]

        triangle_vertex_properties = glGenBuffers(1)
        self.triangle_vertex_properties = triangle_vertex_properties

        glBindBuffer(GL_ARRAY_BUFFER, triangle_vertex_properties)
        glBufferData(GL_ARRAY_BUFFER, vertex_attributes.nbytes,
                     vertex_attributes, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        # Position attribute layout, size,   stride, offset
        glVertexAttribPointer(0, num_pos_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
                              num_properties_per_vertex, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, num_texcoord_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
                              num_properties_per_vertex,
                              ctypes.c_void_p(vertex_attributes.itemsize * num_pos_per_vertex))

        triangle_indices = glGenBuffers(1)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_indices.nbytes,
                     face_indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def update_vao(self, vertex_positions: np.ndarray, texcoord: np.ndarray, face_indices: np.ndarray):
        assert vertex_positions.ndim == 2
        assert vertex_positions.shape[1] == 3
        assert vertex_positions.shape[0] == texcoord.shape[0]
        assert texcoord.shape[1] == 2
        assert face_indices.ndim == 1

        vertex_attributes = np.hstack(
            (vertex_positions, texcoord)).astype(VertextAttribType)

        num_pos_per_vertex = vertex_positions.shape[1]
        num_texcoord_per_vertex = texcoord.shape[1]

        triangle_vao = self.vao

        glBindVertexArray(triangle_vao)

        num_properties_per_vertex = vertex_attributes.shape[1]

        triangle_vertex_properties = self.triangle_vertex_properties

        glBindBuffer(GL_ARRAY_BUFFER, triangle_vertex_properties)
        glBufferData(GL_ARRAY_BUFFER, vertex_attributes.nbytes,
                     vertex_attributes, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        # Position attribute layout, size,   stride, offset
        glVertexAttribPointer(0, num_pos_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
                              num_properties_per_vertex, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, num_texcoord_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
                              num_properties_per_vertex,
                              ctypes.c_void_p(vertex_attributes.itemsize * num_pos_per_vertex))
        glBindVertexArray(0)
