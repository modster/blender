uniform vec3 vert1;
uniform vec3 vert2;
uniform vec3 vert3;
uniform vec3 vert4;

flat out vec4 finalColor;

const int indices[6] = int[6](0,1,2,0,2,3);

void main(void)
{
    vec3 color = vec3(0.2,0.1,1.0);

    vec3[4] verts;
    verts[0] = vert1;
    verts[1] = vert2;
    verts[2] = vert3;
    verts[3] = vert4;

    vec3 pos = verts[indices[gl_VertexID % 6]];
    finalColor = vec4(color,1.0);
    vec3 world_pos = point_object_to_world(pos);
    gl_Position = point_world_to_ndc(world_pos);
}
