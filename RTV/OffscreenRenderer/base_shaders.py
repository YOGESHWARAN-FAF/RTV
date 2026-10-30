VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 position;

out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0f);
    FragPos = vec3(model * vec4(position, 1.0f));
}
"""

GEOMETRY_SHADER = """
#version 330

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in vec3 FragPos [];
out vec3 normal;
out vec3 ex_FragPos;

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
        EmitVertex( );
    }
    
    EndPrimitive( );
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec3 ex_FragPos;
in vec3 normal;


out vec4 color;


//uniform vec3 lightPos;
//uniform vec3 objectColor;



void main()
{
     // Ambient
    //float ambientStrength = 0.2f;
    float ambient = 0.4;
    vec3 lightPos = vec3(10,100,-1000);
    vec3 objectColor = vec3(1,1,1);
    

    // Diffuse
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - ex_FragPos);
    float diffuse = max(dot(norm, lightDir)*0.6, -0.3);



    vec3 result = (ambient + diffuse)* objectColor;
    float gamma = 1.1;
    result = pow(result, vec3(1.0/gamma));
    color = vec4(result, 1);
    //color=vec4(1,0,0,1);
}
"""
