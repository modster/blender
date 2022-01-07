#ifndef USE_GPU_SHADER_CREATE_INFO
uniform mat4 ModelViewProjectionMatrix;

in vec3 pos;
in float size;
#endif

void main()
{
  gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
  gl_PointSize = size;
}
