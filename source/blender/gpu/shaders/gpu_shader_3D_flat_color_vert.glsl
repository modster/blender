#ifndef USE_GPU_SHADER_CREATE_INFO
uniform mat4 ModelViewProjectionMatrix;
#  ifdef USE_WORLD_CLIP_PLANES
uniform mat4 ModelMatrix;
#  endif

in vec3 pos;
<<<<<<< HEAD
#  if defined(USE_COLOR_U32)
in uint color;
#  else
in vec4 color;
#  endif
=======
in vec4 color;
>>>>>>> master

flat out vec4 finalColor;
#endif

void main()
{
  vec4 pos_4d = vec4(pos, 1.0);
  gl_Position = ModelViewProjectionMatrix * pos_4d;
  finalColor = color;

#ifdef USE_WORLD_CLIP_PLANES
  world_clip_planes_calc_clip_distance((ModelMatrix * pos_4d).xyz);
#endif
}
