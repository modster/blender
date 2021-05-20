
IN_OUT VolumeDataInterface
{
  vec3 P_start;
  vec3 P_end;
}
interp;

#ifdef GPU_FRAGMENT_SHADER
GlobalData init_globals(void)
{
  GlobalData volume;
  volume.P = interp.P_start;
  return volume;
}
#endif
