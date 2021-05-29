/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * Copyright 2018, Blender Foundation.
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
  LightCache(const int grid_len,
             const int cube_len,
             const int cube_size,
             const int vis_size,
             const int irr_size[3])
  {
    memset(this, 0, sizeof(*this));

    version = LIGHTCACHE_STATIC_VERSION;
    type = LIGHTCACHE_TYPE_STATIC;
    mips_len = log2_floor_u(cube_size) - min_cube_lod_level;
    vis_res = vis_size;
    ref_res = cube_size;

    cube_data = (LightProbeCache *)MEM_calloc_arrayN(
        cube_len, sizeof(LightProbeCache), "LightProbeCache");
    grid_data = (LightGridCache *)MEM_calloc_arrayN(
        grid_len, sizeof(LightGridCache), "LightGridCache");
    cube_mips = (LightCacheTexture *)MEM_calloc_arrayN(
        mips_len, sizeof(LightCacheTexture), "LightCacheTexture");

    grid_tx.tex_size[0] = irr_size[0];
    grid_tx.tex_size[1] = irr_size[1];
    grid_tx.tex_size[2] = irr_size[2];

    cube_tx.tex_size[0] = ref_res;
    cube_tx.tex_size[1] = ref_res;
    cube_tx.tex_size[2] = cube_len * 6;

    create_reflection_texture();
    create_irradiance_texture();

    if (flag & LIGHTCACHE_NOT_USABLE) {
      /* We could not create the requested textures size. Stop baking and do not use the cache. */
      flag = LIGHTCACHE_INVALID;
      return;
    }

    flag = LIGHTCACHE_UPDATE_WORLD | LIGHTCACHE_UPDATE_CUBE | LIGHTCACHE_UPDATE_GRID;

    for (int mip = 0; mip < mips_len; mip++) {
      GPU_texture_get_mipmap_size(cube_tx.tex, mip + 1, cube_mips[mip].tex_size);
    }
  }

  ~LightCache()
  {
    DRW_TEXTURE_FREE_SAFE(cube_tx.tex);
    MEM_SAFE_FREE(cube_tx.data);
    DRW_TEXTURE_FREE_SAFE(grid_tx.tex);
    MEM_SAFE_FREE(grid_tx.data);

    if (cube_mips) {
      for (int i = 0; i < mips_len; i++) {
        MEM_SAFE_FREE(cube_mips[i].data);
      }
      MEM_SAFE_FREE(cube_mips);
    }

    MEM_SAFE_FREE(cube_data);
    MEM_SAFE_FREE(grid_data);
  }

  int irradiance_cells_per_row_get(void) const
  {
    /* Ambient cube is 3x2px. */
    return grid_tx.tex_size[0] / 3;
  }

  /**
   * Returns dimensions of the irradiance cache texture.
   **/
  static void irradiance_cache_size_get(int visibility_size, int total_samples, int r_size[3])
  {
    /* Compute how many irradiance samples we can store per visibility sample. */
    int irr_per_vis = (visibility_size / irradiance_sample_size_x) *
                      (visibility_size / irradiance_sample_size_y);

    /* The irradiance itself take one layer, hence the +1 */
    int layer_ct = min_ii(irr_per_vis + 1, irradiance_max_pool_layer);

    int texel_ct = (int)ceilf((float)total_samples / (float)(layer_ct - 1));
    r_size[0] = visibility_size *
                max_ii(1, min_ii(texel_ct, (irradiance_max_pool_size / visibility_size)));
    r_size[1] = visibility_size *
                max_ii(1, (texel_ct / (irradiance_max_pool_size / visibility_size)));
    r_size[2] = layer_ct;
  }

  bool validate(const int cube_len,
                const int cube_res,
                const int grid_len,
                const int irr_size[3]) const
  {
    if (!version_check()) {
      return false;
    }

    if ((flag & LIGHTCACHE_INVALID) == 0) {
      /* See if we need the same amount of texture space. */
      if ((ivec3(irr_size) == ivec3(grid_tx.tex_size)) && (grid_len == this->grid_len)) {
        int mip_len = log2_floor_u(cube_res) - min_cube_lod_level;
        if ((cube_res == cube_tx.tex_size[0]) && (cube_len == cube_tx.tex_size[2] / 6) &&
            (cube_len == this->cube_len) && (mip_len == this->mips_len)) {
          return true;
        }
      }
    }
    return false;
  }

 private:
  /**
   * Returns true if the lightcache can be loaded correctly with this version of eevee.
   **/
  bool version_check() const
  {
    switch (type) {
      case LIGHTCACHE_TYPE_STATIC:
        return version == LIGHTCACHE_STATIC_VERSION;
      default:
        return false;
    }
  }

  /**
   * Creates empty texture for reflection data.
   * Returns false on failure and set lightcache as unusable.
   **/
  bool create_reflection_texture(void)
  {
    /* Try to create a cubemap array. */
    cube_tx.tex = GPU_texture_create_cube_array("lightcache_cubemaps",
                                                cube_tx.tex_size[0],
                                                cube_tx.tex_size[2] / 6,
                                                mips_len + 1,
                                                reflection_format,
                                                nullptr);

    if (cube_tx.tex == nullptr) {
      /* Try fallback to 2D array. */
      cube_tx.tex = GPU_texture_create_2d_array("lightcache_cubemaps_fallback",
                                                UNPACK3(cube_tx.tex_size),
                                                mips_len + 1,
                                                reflection_format,
                                                nullptr);
    }

    if (cube_tx.tex != nullptr) {
      GPU_texture_mipmap_mode(cube_tx.tex, true, true);
      /* TODO(fclem) This fixes incomplete texture. Fix the GPU module instead. */
      GPU_texture_generate_mipmap(cube_tx.tex);
    }
    else {
      flag |= LIGHTCACHE_NOT_USABLE;
    }
    return cube_tx.tex != nullptr;
  }

  /**
   * Creates empty texture for irradiance data.
   * Returns false on failure and set lightcache as unusable.
   **/
  bool create_irradiance_texture(void)
  {
    grid_tx.tex = GPU_texture_create_2d_array(
        "lightcache_irradiance", UNPACK3(grid_tx.tex_size), 1, irradiance_format, nullptr);
    if (grid_tx.tex != nullptr) {
      GPU_texture_filter_mode(grid_tx.tex, true);
    }
    else {
      flag |= LIGHTCACHE_NOT_USABLE;
    }
    return grid_tx.tex != nullptr;
  }

  /**
   * Loads a static lightcache data into GPU memory.
   **/
  bool load_static(void)
  {
    /* We use fallback if a texture is not setup and there is no data to restore it. */
    if ((!grid_tx.tex && !grid_tx.data) || !grid_data || (!cube_tx.tex && !cube_tx.data) ||
        !cube_data) {
      return false;
    }
    /* If cache is too big for this GPU. */
    if (cube_tx.tex_size[2] > GPU_max_texture_layers()) {
      return false;
    }

    if (grid_tx.tex == nullptr) {
      if (create_irradiance_texture()) {
        GPU_texture_update(grid_tx.tex, GPU_DATA_UBYTE, grid_tx.data);
      }
    }

    if (cube_tx.tex == nullptr) {
      if (create_reflection_texture()) {
        for (int mip = 0; mip <= mips_len; mip++) {
          const void *data = (mip == 0) ? cube_tx.data : cube_mips[mip - 1].data;
          GPU_texture_update_mipmap(cube_tx.tex, mip, GPU_DATA_10_11_11_REV, data);
        }
      }
    }
    return true;
  }

  MEM_CXX_CLASS_ALLOC_FUNCS("EEVEE:LightCache")
};

}  // namespace blender::eevee
