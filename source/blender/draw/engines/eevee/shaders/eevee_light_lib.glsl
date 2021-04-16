
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

int findLSB(uint x)
{
  for (uint i = 0u; i < 32u; i++) {
    if ((x & (1u << i)) != 0u) {
      return int(i);
    }
  }
  return -1;
}

#define LIGHT_FOREACH_BEGIN(_clusters, _lights, _light) \
  { \
    uvec4 _tile_data = _clusters.cells[0]; \
    int _ofs = 0; \
    while (_tile_data != uvec4(0)) { \
      if (_tile_data.x == 0u) { \
        _tile_data = _tile_data.yzwx; \
        _ofs += 32; \
      } \
      int _i = findLSB(_tile_data.x); \
      if (_i == -1) { \
        continue; \
      } \
      _tile_data.x &= ~(1u << _i); \
      LightData _light = _lights[_i + _ofs];

#define LIGHT_FOREACH_END \
  } \
  }