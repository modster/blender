
/**
 * As input to the tracing function, direction is premultiplied by its maximum length.
 * As output, direction is scaled to hit point or to latest step.
 */
struct Ray {
  vec3 origin;
  vec3 direction;
};

/**
 * Screenspace ray ([0..1] "uv" range) where direction is normalize to be as small as one
 * full-resolution pixel. The ray is also clipped to all frustum sides.
 */
struct ScreenSpaceRay {
  vec4 origin;
  vec4 direction;
  float max_time;
};
