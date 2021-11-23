
/**
 * Sort the lights by their Z distance to the camera.
 * Outputs ordered light buffer and associated zbins.
 * We split the work in CULLING_BATCH_SIZE and iterate to cover all zbins.
 * One thread process one Light entity.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_light_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(local_size_x = CULLING_BATCH_SIZE) in;

layout(std430, binding = 0) readonly restrict buffer lights_buf
{
  LightData lights[];
};

layout(std430, binding = 1) restrict buffer culling_buf
{
  CullingData culling;
};

layout(std430, binding = 2) readonly restrict buffer key_buf
{
  uint keys[];
};

layout(std430, binding = 3) writeonly restrict buffer out_zbins_buf
{
  CullingZBin out_zbins[];
};

layout(std430, binding = 4) writeonly restrict buffer out_items_buf
{
  LightData out_lights[];
};

void main()
{
  uint src_index = gl_GlobalInvocationID.x;
  bool valid_thread = true;

  if (src_index >= culling.visible_count) {
    /* Do not return because we use barriers later on (which need uniform control flow).
     * Just process the same last item but avoid insertion. */
    src_index = culling.visible_count - 1;
    valid_thread = false;
  }

  uint key = keys[src_index];
  LightData light = lights[key];

  if (!culling.enable_specular) {
    light.specular_power = 0.0;
  }

  int index = 0;
  int contenders = 0;

  /* TODO(fclem): Sun lights are polutting the zbins with no reasons. Better bypass culling. */
  vec3 lP = (light.type == LIGHT_SUN) ? cameraPos : light._position;
  float radius = (light.type == LIGHT_SUN) ? ViewFar * 2.0 : light.influence_radius_max;
  float z_dist = dot(cameraForward, lP) - dot(cameraForward, cameraPos);

  int z_min = clamp(culling_z_to_zbin(culling, z_dist + radius), 0, CULLING_ZBIN_COUNT - 1);
  int z_max = clamp(culling_z_to_zbin(culling, z_dist - radius), 0, CULLING_ZBIN_COUNT - 1);

  if (!valid_thread) {
    /* Do not register invalid threads. */
    z_max = z_min - 1;
  }

  /* Fits the limit of 32KB. */
  shared int zbin_max[CULLING_ZBIN_COUNT];
  shared int zbin_min[CULLING_ZBIN_COUNT];
  /* Compilers do not release shared memory from early declaration.
   * So we are forced to reuse the same variables in another form. */
#define z_dists zbin_max
#define contender_table zbin_min

  /**
   * Find how many values are before the local value.
   * This finds the first possible destination index.
   */
  z_dists[gl_LocalInvocationID.x] = floatBitsToInt(z_dist);
  barrier();

  const uint i_start = gl_WorkGroupID.x * CULLING_BATCH_SIZE;
  uint i_max = min(CULLING_BATCH_SIZE, culling.visible_count - i_start);
  for (uint i = 0; i < i_max; i++) {
    float ref = intBitsToFloat(z_dists[i]);
    if (ref > z_dist) {
      index++;
    }
    else if (ref == z_dist) {
      contenders++;
    }
  }

  atomicExchange(contender_table[index], contenders);
  barrier();

  if (valid_thread) {
    /**
     * For each clashing index (where two lights have exactly the same z distances)
     * we use an atomic counter to know how much to offset from the disputed index.
     */
    index += atomicAdd(contender_table[index], -1) - 1;
    index += int(i_start);
    out_lights[index] = light;
  }

  const uint iter = uint(CULLING_ZBIN_COUNT / CULLING_BATCH_SIZE);
  const uint zbin_local = gl_LocalInvocationID.x * iter;
  const uint zbin_global = gl_WorkGroupID.x * CULLING_ZBIN_COUNT + zbin_local;

  for (uint i = 0u, l = zbin_local; i < iter; i++, l++) {
    zbin_max[l] = 0x0000;
    zbin_min[l] = 0xFFFF;
  }
  barrier();

  /* Register to Z bins. */
  for (int z = z_min; z <= z_max; z++) {
    atomicMin(zbin_min[z], index);
    atomicMax(zbin_max[z], index);
  }
  barrier();

  /* Write result to zbins buffer. */
  for (uint i = 0u, g = zbin_global, l = zbin_local; i < iter; i++, g++, l++) {
    /* Pack min & max into 1 uint. */
    out_zbins[g] = (uint(zbin_max[l]) << 16u) | uint(zbin_min[l]);
  }
}
