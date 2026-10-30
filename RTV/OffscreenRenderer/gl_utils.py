from typing import Tuple
import ctypes
import os
import platform
import sys

import numpy as np

try:
    if platform.system() == "Windows":
        egl = None
    else:
        import OpenGL.EGL as egl
        from OpenGL.raw.EGL.EXT.platform_device import EGL_PLATFORM_DEVICE_EXT
        from OpenGL.EGL.EXT.device_base import egl_get_devices
except Exception:
    egl = None

from OpenGL import error
from OpenGL.GL import *


def create_initialized_headless_egl_display():
    """Creates an initialized EGL display directly on a device.
    """
    if egl is None:
        return None
    for device in egl_get_devices():
        display = egl.eglGetPlatformDisplayEXT(
            EGL_PLATFORM_DEVICE_EXT, device, None)

        if display != egl.EGL_NO_DISPLAY and egl.eglGetError() == egl.EGL_SUCCESS:
            # `eglInitialize` may or may not raise an exception on failure depending
            # on how PyOpenGL is configured. We therefore catch a `GLError` and also
            # manually check the output of `eglGetError()` here.
            try:
                initialized = egl.eglInitialize(display, None, None)
            except error.GLError:
                pass
            else:
                if initialized == egl.EGL_TRUE and egl.eglGetError() == egl.EGL_SUCCESS:
                    return display
    return egl.EGL_NO_DISPLAY


def create_opengl_context(surface_size: Tuple[int, int]):
    """Create offscreen OpenGL context and make it current.

    Users are expected to directly use EGL API in case more advanced

    context management is required.

    Args:
        surface_size: (width, height), size of the offscreen rendering surface.

    """
    width, height = surface_size

    # 1. Try GLFW first: completely thread-safe and works on any worker thread on Windows/Linux
    try:
        import glfw
        if glfw.init():
            glfw.window_hint(glfw.VISIBLE, False)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            window = glfw.create_window(width, height, "Offscreen", None, None)
            if window:
                glfw.make_context_current(window)
                return None, window, window
    except Exception:
        pass

    if egl is None:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QOpenGLContext, QOffscreenSurface, QSurfaceFormat
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        ctx.create()
        surf = QOffscreenSurface()
        surf.setFormat(fmt)
        surf.create()
        ctx.makeCurrent(surf)
        return None, surf, ctx

    egl_display = create_initialized_headless_egl_display()

    if egl_display is None or (egl is not None and egl_display == egl.EGL_NO_DISPLAY):
        if egl is not None:
            try:
                def_disp = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
                if def_disp != egl.EGL_NO_DISPLAY:
                    maj, min_ = egl.EGLint(), egl.EGLint()
                    if egl.eglInitialize(def_disp, ctypes.pointer(maj), ctypes.pointer(min_)) == egl.EGL_TRUE:
                        egl_display = def_disp
            except Exception:
                pass

    if egl_display is None or (egl is not None and egl_display == egl.EGL_NO_DISPLAY):
        # Fallback to Qt offscreen if EGL device display is unavailable
        try:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QOpenGLContext, QOffscreenSurface, QSurfaceFormat
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            fmt = QSurfaceFormat()
            fmt.setDepthBufferSize(24)
            fmt.setStencilBufferSize(8)
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.CoreProfile)
            ctx = QOpenGLContext()
            ctx.setFormat(fmt)
            ctx.create()
            surf = QOffscreenSurface()
            surf.setFormat(fmt)
            surf.create()
            ctx.makeCurrent(surf)
            return None, surf, ctx
        except Exception as e:
            raise RuntimeError(f"Cannot initialize a headless OpenGL context (EGL and Qt fallback failed): {e}")

    major, minor = egl.EGLint(), egl.EGLint()

    egl.eglInitialize(egl_display, ctypes.pointer(
        major), ctypes.pointer(minor))

    config_attribs = [
        egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT, egl.EGL_BLUE_SIZE, 8,
        egl.EGL_GREEN_SIZE, 8, egl.EGL_RED_SIZE, 8, egl.EGL_DEPTH_SIZE, 24,
        egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT, egl.EGL_NONE
    ]
    config_attribs = (egl.EGLint * len(config_attribs))(*config_attribs)

    num_configs = egl.EGLint()
    egl_cfg = egl.EGLConfig()

    egl.eglChooseConfig(egl_display, config_attribs, ctypes.pointer(egl_cfg), 1,
                        ctypes.pointer(num_configs))

    width, height = surface_size

    pbuffer_attribs = [
        egl.EGL_WIDTH,
        width,
        egl.EGL_HEIGHT,
        height,
        egl.EGL_NONE,
    ]
    pbuffer_attribs = (egl.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
    egl_surf = egl.eglCreatePbufferSurface(
        egl_display, egl_cfg, pbuffer_attribs)

    egl.eglBindAPI(egl.EGL_OPENGL_API)

    egl_context = egl.eglCreateContext(
        egl_display, egl_cfg, egl.EGL_NO_CONTEXT, None)

    egl.eglMakeCurrent(egl_display, egl_surf, egl_surf, egl_context)

    return egl_display, egl_surf, egl_context


def init_frame_buffer(frame_image: np.ndarray, num_samples: int = 16):
    """Create an FBO and assign a texture buffer to it for the purpose of offscreen rendering to the texture buffer
    """
    samples = min(num_samples, GL_SAMPLES - 1)

    width = frame_image.shape[1]
    height = frame_image.shape[0]

    if frame_image.shape[-1] == 3:
        texture_type = GL_RGB
        internal_texture_type = GL_RGB8
    elif frame_image.shape[-1] == 4:
        texture_type = GL_RGBA
        internal_texture_type = GL_RGBA8

    render_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, render_texture)
    glTexImage2D(GL_TEXTURE_2D, 0, internal_texture_type, width,
                 height, 0, texture_type, GL_UNSIGNED_BYTE, None)

    glBindTexture(GL_TEXTURE_2D, 0)

    render_frame_object = glGenFramebuffers(1)

    glBindFramebuffer(GL_FRAMEBUFFER, render_frame_object)

    # Create color render buffer
    color_buffer = glGenRenderbuffers(1)

    glBindRenderbuffer(GL_RENDERBUFFER, color_buffer)
    glRenderbufferStorage(
        GL_RENDERBUFFER, internal_texture_type, width, height)
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_buffer)

    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render_texture, 0)

    depth_buffer = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height)
    glBindRenderbuffer(GL_RENDERBUFFER,0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth_buffer)

    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    # Sees if your GPU can handle the FBO configuration defined above
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(
            "Framebuffer binding failed, probably because your GPU does not support this FBO configuration.")

    glBindFramebuffer(GL_FRAMEBUFFER, 0)



    return render_frame_object
