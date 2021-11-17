
/**
 * Debugging drawing library
 *
 * Quick way to draw debug geometry. All input should be in world space and
 * will be rendered in the default view. No additional setup required.
 **/
#define DEBUG_DRAW

/* Keep in sync with buffer creation. */
#define DEBUG_VERT_MAX 16 * 4096

struct DebugVert {
  vec3 pos;
  uint color;
};

layout(std430, binding = 7) restrict buffer debugBuf
{
  /** Start the buffer with a degenerate vertice. */
  uint _pad0;
  uint _pad1;
  uint _pad2;
  uint v_count;
  DebugVert verts[];
}
drw_debug_verts;

bool drw_debug_draw_enable = true;

uint drw_debug_color_pack(vec4 color)
{
  color = clamp(color, 0.0, 1.0);
  uint result = 0;
  result |= uint(color.x * 255.0) << 0u;
  result |= uint(color.y * 255.0) << 8u;
  result |= uint(color.z * 255.0) << 16u;
  result |= uint(color.w * 255.0) << 24u;
  return result;
}

void drw_debug_line_do(inout uint vertid, vec3 v1, vec3 v2, uint color)
{
  drw_debug_verts.verts[vertid++] = DebugVert(v1, color);
  drw_debug_verts.verts[vertid++] = DebugVert(v2, color);
}

void drw_debug_line(vec3 v1, vec3 v2, vec4 color)
{
  if (!drw_debug_draw_enable) {
    return;
  }
  const uint vneeded = 2;
  uint vertid = atomicAdd(drw_debug_verts.v_count, vneeded);
  if (vertid + vneeded < DEBUG_VERT_MAX + 1) {
    drw_debug_line_do(vertid, v1, v2, drw_debug_color_pack(color));
  }
}

void drw_debug_quad(vec3 v1, vec3 v2, vec3 v3, vec3 v4, vec4 color)
{
  if (!drw_debug_draw_enable) {
    return;
  }
  const uint vneeded = 8;
  uint vertid = atomicAdd(drw_debug_verts.v_count, vneeded);
  if (vertid + vneeded < DEBUG_VERT_MAX + 1) {
    uint pcolor = drw_debug_color_pack(color);
    drw_debug_line_do(vertid, v1, v2, pcolor);
    drw_debug_line_do(vertid, v2, v3, pcolor);
    drw_debug_line_do(vertid, v3, v4, pcolor);
    drw_debug_line_do(vertid, v4, v1, pcolor);
  }
}

/* Draw an octahedron. */
void drw_debug_point(vec3 p, float radius, vec4 color)
{
  if (!drw_debug_draw_enable) {
    return;
  }
  vec3 c = vec3(radius, -radius, 0);
  vec3 v1 = p + c.xzz;
  vec3 v2 = p + c.zxz;
  vec3 v3 = p + c.yzz;
  vec3 v4 = p + c.zyz;
  vec3 v5 = p + c.zzx;
  vec3 v6 = p + c.zzy;

  const uint vneeded = 12 * 2;
  uint vertid = atomicAdd(drw_debug_verts.v_count, vneeded);
  if (vertid + vneeded < DEBUG_VERT_MAX + 1) {
    uint pcolor = drw_debug_color_pack(color);
    drw_debug_line_do(vertid, v1, v2, pcolor);
    drw_debug_line_do(vertid, v2, v3, pcolor);
    drw_debug_line_do(vertid, v3, v4, pcolor);
    drw_debug_line_do(vertid, v4, v1, pcolor);
    drw_debug_line_do(vertid, v1, v5, pcolor);
    drw_debug_line_do(vertid, v2, v5, pcolor);
    drw_debug_line_do(vertid, v3, v5, pcolor);
    drw_debug_line_do(vertid, v4, v5, pcolor);
    drw_debug_line_do(vertid, v1, v6, pcolor);
    drw_debug_line_do(vertid, v2, v6, pcolor);
    drw_debug_line_do(vertid, v3, v6, pcolor);
    drw_debug_line_do(vertid, v4, v6, pcolor);
  }
}

void drw_debug_matrix(mat4 mat, vec4 color, const bool do_project)
{
  vec4 p[8] = vec4[8](vec4(-1, -1, -1, 1),
                      vec4(1, -1, -1, 1),
                      vec4(1, 1, -1, 1),
                      vec4(-1, 1, -1, 1),
                      vec4(-1, -1, 1, 1),
                      vec4(1, -1, 1, 1),
                      vec4(1, 1, 1, 1),
                      vec4(-1, 1, 1, 1));
  for (int i = 0; i < 8; i++) {
    p[i] = mat * p[i];
    if (do_project) {
      p[i].xyz /= p[i].w;
    }
  }
  drw_debug_quad(p[0].xyz, p[1].xyz, p[2].xyz, p[3].xyz, color);
  drw_debug_line(p[0].xyz, p[4].xyz, color);
  drw_debug_line(p[1].xyz, p[5].xyz, color);
  drw_debug_line(p[2].xyz, p[6].xyz, color);
  drw_debug_line(p[3].xyz, p[7].xyz, color);
  drw_debug_quad(p[4].xyz, p[5].xyz, p[6].xyz, p[7].xyz, color);
}
