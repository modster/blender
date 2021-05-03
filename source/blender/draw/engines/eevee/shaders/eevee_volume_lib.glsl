
IN_OUT VolumeDataInterface
{
  vec3 P_start;
  vec3 P_end;
}
interp;

struct VolumeData {
  /** World position. */
  vec3 P;
};

#ifdef GPU_FRAGMENT_SHADER
VolumeData init_from_interp(void)
{
  VolumeData volume;
  volume.P = interp.P_start;
  return volume;
}
#endif