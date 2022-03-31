/* A shorthand for textureSize with a zero LOD. */
ivec2 texture_size(sampler2D sampler)
{
  return textureSize(sampler, 0);
}

/* A shorthand for texelFetch with zero LOD and bounded access clamped to border. */
vec4 load_texture(sampler2D sampler, ivec2 xy)
{
  ivec2 texture_bounds = texture_size(sampler) - ivec2(1);
  return texelFetch(sampler, min(xy, texture_bounds), 0);
}
