
float wang_hash_noise(uint s)
{
  s = (s ^ 61u) ^ (s >> 16u);
  s *= 9u;
  s = s ^ (s >> 4u);
  s *= 0x27d4eb2du;
  s = s ^ (s >> 15u);

  return fract(float(s) / 4294967296.0);
}

void node_hair_info(float hair_length,
                    out float is_strand,
                    out float intercept,
                    out float out_length,
                    out float thickness,
                    out vec3 tangent,
                    out float random)
{
  is_strand = float(g_data.is_strand);
  intercept = g_data.hair_time;
  thickness = g_data.hair_thickness;
  out_length = hair_length;
  tangent = normalize(interp.N);
  /* TODO: could be precomputed per strand instead. */
  random = wang_hash_noise(uint(g_data.hair_strand_id));
}
