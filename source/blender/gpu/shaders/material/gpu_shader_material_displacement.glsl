void node_displacement_object(float height, float midlevel, float scale, vec3 N, out vec3 result)
{
  N = normal_world_to_object(N);
  result = (height - midlevel) * scale * normalize(N);
  result = normal_object_to_world(result);
}

void node_displacement_world(float height, float midlevel, float scale, vec3 N, out vec3 result)
{
  result = (height - midlevel) * scale * normalize(N);
}
