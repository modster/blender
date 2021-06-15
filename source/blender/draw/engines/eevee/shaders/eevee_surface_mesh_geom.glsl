
/**
 * Optional geometry shader stage to compute barycentric coords
 * Only needed / compatible with mesh or gpencil geometry.
 * Main is generated in eevee_shader.cc to avoid compilation issue on some drivers.
 */

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

vec3 calc_barycentric_distances(vec3 pos0, vec3 pos1, vec3 pos2)
{
  vec3 edge21 = pos2 - pos1;
  vec3 edge10 = pos1 - pos0;
  vec3 edge02 = pos0 - pos2;
  vec3 d21 = normalize(edge21);
  vec3 d10 = normalize(edge10);
  vec3 d02 = normalize(edge02);

  vec3 dists;
  float d = dot(d21, edge02);
  dists.x = sqrt(dot(edge02, edge02) - d * d);
  d = dot(d02, edge10);
  dists.y = sqrt(dot(edge10, edge10) - d * d);
  d = dot(d10, edge21);
  dists.z = sqrt(dot(edge21, edge21) - d * d);
  return dists;
}

vec2 calc_barycentric_co(int vertid)
{
  return vec2((vertid % 3) == 0, (vertid % 3) == 1);
}
