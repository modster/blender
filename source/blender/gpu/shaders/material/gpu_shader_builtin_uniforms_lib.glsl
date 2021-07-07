
#ifdef BUILTIN_UNIFORMS_BUFFER
#  if BUILTIN_UNIFORMS_BUFFER == 1

layout(std140) uniform shaderBlock
{
  mat4 ModelMatrix;
  mat4 ModelViewProjectionMatrix;
  vec4 color;
  vec4 WorldClipPlanes[6];
  bool srgbTransform;
};

#  endif
#endif