void node_vector_displacement_tangent(vec3 vector,
                                      float midlevel,
                                      float scale,
                                      vec4 tangent,
                                      vec3 normal,
                                      mat4 obmat,
                                      mat4 viewmat,
                                      out vec3 result)
{
  vec3 oN = normalize(normal_world_to_object(normal));
  vec3 oT = normalize(normal_world_to_object(tangent.xyz));
  vec3 oB = tangent.w * normalize(cross(oN, oT));

  result = (vector - midlevel) * scale;
  result = result.x * oT + result.y * oN + result.z * oB;
  result = transform_point(ModelMatrix, result);
}

void node_vector_displacement_object(
    vec3 vector, float midlevel, float scale, mat4 obmat, out vec3 result)
{
  result = (vector - midlevel) * scale;
  result = transform_point(ModelMatrix, result);
}

void node_vector_displacement_world(vec3 vector, float midlevel, float scale, out vec3 result)
{
  result = (vector - midlevel) * scale;
}
