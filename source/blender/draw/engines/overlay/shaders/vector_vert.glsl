
uniform vec3 objPosition;

flat out vec4 finalColor;

const vec3 verts[5] = vec3[5](vec3(0.0, 0.0, 0.0),
                                vec3(0.0, 0.0, 1.8),
                                 vec3(0.0, -0.0866, 1.8),
                                  vec3(0.0, 0.0, 2.0),
                                   vec3(0.0, 0.0866, 1.8));

const int indices[6] = int[6](0,3,3,2,3,4);

void main() {
    GPU_INTEL_VERTEX_SHADER_WORKAROUND
    vec3 color = vec3(1.0,0.0,1.0);
    vec3 pos = vec3(0.0,0.0,0.0);
    pos += objPosition;
    pos += verts[indices[gl_VertexID % 6]];
    finalColor = vec4(color, 3.0);
    vec3 world_pos = point_object_to_world(pos);
    gl_Position = point_world_to_ndc(world_pos);
}
