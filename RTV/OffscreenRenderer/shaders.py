VERTEX_SHADER = """
#version 330 core
uniform mat4   model;         // Model matrix
uniform mat4   view;          // View matrix
uniform mat4   projection;    // Projection matrix
layout (location=0) in vec3 a_position;      // Vertex position
layout (location=1) in  vec2 a_texcoord;      // Vertex texture coordinates
out vec2   v_texcoord;      // Interpolated fragment texture coordinates (out)
void main()
{
    // Assign varying variables
    v_texcoord  = a_texcoord;
    // Final position
    gl_Position = projection * view * model * vec4(a_position,1.0);
    //gl_Position.z = 0.0;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2      v_texcoord; // Interpolated fragment texture coordinates (in)
out vec4 color;
// // // // // out float gl_FragDepth;
uniform sampler2D u_texture;  // Texture 
void main()
{
    // Get texture color
    vec4 t_color = texture(u_texture, v_texcoord);
    color = t_color;
    // Final color
    //gl_FragColor = t_color;//vec4(1.0,0,0,1.0);
}
"""
