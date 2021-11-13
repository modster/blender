#define BLENDER_ZMAX 10000.0

void node_composite_map_range(float value,
                              float from_min,
                              float from_max,
                              float to_min,
                              float to_max,
                              const float should_clamp,
                              out float result)
{
  if (abs(from_max - from_min) < 1e-6) {
    result = 0.0;
  }
  else {
    if (value >= -BLENDER_ZMAX && value <= BLENDER_ZMAX) {
      result = (value - from_min) / (from_max - from_min);
      result = to_min + result * (to_max - to_min);
    }
    else if (value > BLENDER_ZMAX) {
      result = to_max;
    }
    else {
      result = to_min;
    }

    if (should_clamp != 0.0) {
      if (to_max > to_min) {
        result = clamp(result, to_min, to_max);
      }
      else {
        result = clamp(result, to_max, to_min);
      }
    }
  }
}
