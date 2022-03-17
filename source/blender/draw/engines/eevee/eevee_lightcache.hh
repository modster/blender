/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2018 Blender Foundation.
 */

/** \file
 * \ingroup eevee
 */

#pragma once

#include "MEM_guardedalloc.h"

#include "DNA_lightprobe_types.h"

#include "GPU_capabilities.h"

namespace blender::eevee {

/**
 * Wrapper to blender lightcache structure.
 * Used to define methods for the light cache.
 **/
struct LightCache : public ::LightCache {
 private:
  constexpr static int min_cube_lod_level = 3;
  /* Rounded to nearest PowerOfTwo */
  constexpr static int irradiance_sample_size_x = 4; /* 3 in reality */
  constexpr static int irradiance_sample_size_y = 2;
  /* Manually encoded as RGBM. Also encodes visibility. */
  constexpr static eGPUTextureFormat irradiance_format = GPU_RGBA8;
  /* OpenGL 3.3 core requirement, can be extended but it's already very big */
  constexpr static int irradiance_max_pool_layer = 256;
  constexpr static int irradiance_max_pool_size = 1024;
  constexpr static int max_irradiance_samples =
      (irradiance_max_pool_size / irradiance_sample_size_x) *
      (irradiance_max_pool_size / irradiance_sample_size_y);

  constexpr static eGPUTextureFormat reflection_format = GPU_R11F_G11F_B10F;

 public:
  LightCache(const int cube_len,
             const int grid_len,
             const int cube_size,
             const int vis_size,
             const int irr_size[3]);

  ~LightCache();

  static void irradiance_cache_size_get(int visibility_size, int total_samples, int r_size[3]);
  static void update_info(SceneEEVEE *eevee);

  int irradiance_cells_per_row_get(void) const;

  bool load(void);

  bool validate(const int cube_len,
                const int cube_res,
                const int grid_len,
                const int irr_size[3]) const;

  uint memsize_get(void) const;
  int64_t irradiance_sample_count(void) const;

  void readback_irradiance(void);
  void readback_reflections(void);

  eGPUTextureFormat irradiance_format_get(void) const
  {
    return irradiance_format;
  }

  eGPUTextureFormat reflection_format_get(void) const
  {
    return reflection_format;
  }

 private:
  bool version_check(void) const;
  bool can_be_saved(void) const;
  bool load_static(void);

  bool create_reflection_texture(void);
  bool create_irradiance_texture(void);

  MEM_CXX_CLASS_ALLOC_FUNCS("EEVEE:LightCache")
};

}  // namespace blender::eevee
