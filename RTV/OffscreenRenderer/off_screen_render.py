from OpenGL.GL import *
import numpy as np
from PIL import Image

VertextAttribType = np.float32
IndexType = np.uint32
OpenglVertexAttrType = GL_FLOAT
OpenglTriangleIndexType = GL_UNSIGNED_INT


def generate_vao(vertex_positions: np.ndarray, texcoord: np.ndarray, face_indices: np.ndarray):
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

    glBindVertexArray(triangle_vao)

    num_properties_per_vertex = vertex_attributes.shape[1]

    triangle_vertex_properties = glGenBuffers(1)

    glBindBuffer(GL_ARRAY_BUFFER, triangle_vertex_properties)
    glBufferData(GL_ARRAY_BUFFER, vertex_attributes.nbytes,
                 vertex_attributes, GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(0)
    # Position attribute layout, size,   stride, offset
    glVertexAttribPointer(0, num_pos_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
                          num_properties_per_vertex, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, num_texcoord_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
                          num_properties_per_vertex, ctypes.c_void_p(vertex_attributes.itemsize * num_pos_per_vertex))

    triangle_indices = glGenBuffers(1)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_indices)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_indices.nbytes,
                 face_indices, GL_STATIC_DRAW)

    glBindVertexArray(0)

    return triangle_vao


def set_shader_params(shader, matrix_model=None, matrix_view=None, matrix_proj=None):
    matrix_proj_loc = glGetUniformLocation(shader, "projection")
    if matrix_proj is None:
        matrix_proj = np.eye(4)
    glUniformMatrix4fv(matrix_proj_loc, 1, GL_FALSE, matrix_proj)

    matrix_model_loc = glGetUniformLocation(shader, "model")
    if matrix_model is None:
        matrix_model = np.eye(4)
    glUniformMatrix4fv(matrix_model_loc, 1, GL_FALSE, matrix_model)

    matrix_view_loc = glGetUniformLocation(shader, "view")
    if matrix_view is None:
        matrix_view = np.eye(4)
    glUniformMatrix4fv(matrix_view_loc, 1, GL_FALSE, matrix_view)

import time

def read_texture(filename):
    img = Image.open(filename)
    img_np = np.array(img)
    img_np = img_np[:, :, [2,1,0]]
    #img_np = np.ones_like(img_np)*123
    img = Image.fromarray(img_np,"RGB")
    img_data = np.array(list(img.getdata()), np.uint8)
    textID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textID)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    try:
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    except Exception:
        pass
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.size[0], img.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glBindTexture(GL_TEXTURE_2D, 0)
    t1=time.time()
    return textID


def render(gl_primitive_type, num_elements: int, texID=None):
    if texID is not None:
        try:
            glEnable(GL_TEXTURE_2D)
        except Exception:
            pass
        glBindTexture(GL_TEXTURE_2D, texID)
    glDrawElements(gl_primitive_type, num_elements, OpenglTriangleIndexType, ctypes.c_void_p(0))
