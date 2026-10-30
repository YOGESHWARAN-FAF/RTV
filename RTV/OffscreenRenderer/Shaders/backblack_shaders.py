VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 position;
layout (location=1) in  vec2 a_texcoord;      // Vertex texture coordinates
out vec2   v_texcoord;      // Interpolated fragment texture coordinates (out)
out vec3 FragPos;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0f);
    FragPos = vec3(model * vec4(position, 1.0f));
    v_texcoord = a_texcoord;
}
"""

GEOMETRY_SHADER = """
#version 330

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in vec2 v_texcoord [];
in vec3 FragPos [];
out vec3 normal;
out vec3 ex_FragPos;
out vec2 ex_textcoord;

void main( void )
{
    //vec3 a = ( gl_in[1].gl_Position - gl_in[0].gl_Position ).xyz;
    vec3 a = FragPos[1]-FragPos[0];
    //vec3 b = ( gl_in[2].gl_Position - gl_in[0].gl_Position ).xyz;
    vec3 b = FragPos[2]-FragPos[0];
    vec3 N = -normalize( cross( b, a ) );

    for( int i=0; i<gl_in.length( ); ++i )
    {
        gl_Position = gl_in[i].gl_Position;
        normal = N;
        ex_FragPos=FragPos[i];
        ex_textcoord=v_texcoord[i];
        EmitVertex( );
    }
    
    EndPrimitive( );
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec3 ex_FragPos;
in vec3 normal;
in vec2 ex_textcoord; // Interpolated fragment texture coordinates (in)


out vec4 color;


//uniform vec3 lightPos;
//uniform vec3 objectColor;
uniform sampler2D u_texture;  // Texture 

void main()
{
    vec4 t_color = texture(u_texture, ex_textcoord);
    if(normal.z<0)
    color = t_color;
    else
    color=vec4(0,0,0,1.0);
}
"""
