void world_normals_get(out vec3 N)
{
  N = FrontFacing ? g_data.N : -g_data.N;
}
