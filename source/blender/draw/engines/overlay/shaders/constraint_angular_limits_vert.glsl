uniform vec4 mat_vecs[4];
uniform float offset;
uniform float angle;
uniform vec3 color;

flat out vec4 finalColor;

const float PI = 3.141592653589;
const float segment_angular_width = PI * 10/180;

int n_segments = int(floor(angle/segment_angular_width));

int indices[74];

void main(void)
{
    for(int i=0; i<=(n_segments); i++) {
        indices[2*i] = i;
        indices[2*i+1] = i+1;
    }
    indices[2*(n_segments+1)] = n_segments+1;
    indices[2*(n_segments+1)+1] = 0;

  vec4 p[37];
  p[0] = vec4(0.0, 0.0, 0.0, 1.0);
  for (int i = 0; i < n_segments; i++) {
    p[i+1] = vec4(cos((angle * i / n_segments)+ offset), sin((angle * i / n_segments)+offset), 0.0, 1.0);
  }
   p[n_segments+1] = vec4(cos(angle + offset), sin(angle + offset), 0.0, 1.0);

  int ind = indices[gl_VertexID%74];

  vec4 transform;
  mat4 mat;
  mat = mat4(mat_vecs[0], mat_vecs[1], mat_vecs[2], mat_vecs[3]);
  transform = mat * p[ind];

  vec3 pos = transform.xyz;
  finalColor = vec4(color, 1.0);
  vec3 world_pos = point_object_to_world(pos);
  gl_Position = point_world_to_ndc(world_pos);
}
