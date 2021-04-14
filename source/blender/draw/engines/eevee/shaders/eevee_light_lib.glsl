

#define LIGHT_FOREACH_BEGIN(_scene, _lights, _light) \
  { \
    for (uint i = 0u; i < _scene.light_count; i++) { \
      LightData _light = _lights[i];

#define LIGHT_FOREACH_END \
  } \
  }