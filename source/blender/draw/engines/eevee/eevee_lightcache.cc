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
 * Copyright 2016, Blender Foundation.
 */

/** \file
 * \ingroup draw_engine
 *
 * Eevee's indirect lighting cache.
 */

#include "DRW_render.h"

#include "BKE_global.h"

#include "BLI_endian_switch.h"
#include "BLI_span.hh"
#include "BLI_threads.h"

#include "DEG_depsgraph_build.h"
#include "DEG_depsgraph_query.h"

#include "BKE_object.h"

#include "DNA_collection_types.h"
#include "DNA_lightprobe_types.h"

#include "PIL_time.h"

#include "eevee_instance.hh"
#include "eevee_lightcache.h"

#include "GPU_capabilities.h"
#include "GPU_context.h"

#include "WM_api.h"
#include "WM_types.h"

#include "BLO_read_write.h"

#include "wm_window.h"

/* TODO should be replace by a more elegant alternative. */
extern void DRW_opengl_context_enable(void);
extern void DRW_opengl_context_disable(void);

extern void DRW_opengl_render_context_enable(void *re_gl_context);
extern void DRW_opengl_render_context_disable(void *re_gl_context);
extern void DRW_gpu_render_context_enable(void *re_gpu_context);
extern void DRW_gpu_render_context_disable(void *re_gpu_context);

typedef struct EEVEE_LightBake {
  int pad;
} EEVEE_LightBake;

/* -------------------------------------------------------------------- */
/** \name Light Cache
 * \{ */

namespace blender::eevee {

LightCache::LightCache(const int grid_len,
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

LightCache::~LightCache()
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

int LightCache::irradiance_cells_per_row_get(void) const
{
  /* Ambient cube is 3x2px. */
  return grid_tx.tex_size[0] / 3;
}

/**
 * Returns dimensions of the irradiance cache texture.
 **/
void LightCache::irradiance_cache_size_get(int visibility_size, int total_samples, int r_size[3])
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

bool LightCache::validate(const int cube_len,
                          const int cube_res,
                          const int grid_len,
                          const int irr_size[3]) const
{
  if (!version_check()) {
    return false;
  }
  if ((flag & (LIGHTCACHE_INVALID | LIGHTCACHE_NOT_USABLE)) != 0) {
    return false;
  }
  /* See if we need the same amount of texture space. */
  if ((ivec3(irr_size) == ivec3(grid_tx.tex_size)) && (grid_len == this->grid_len)) {
    int mip_len = log2_floor_u(cube_res) - min_cube_lod_level;
    if ((cube_res == cube_tx.tex_size[0]) && (cube_len == cube_tx.tex_size[2] / 6) &&
        (cube_len == this->cube_len) && (mip_len == this->mips_len)) {
      return true;
    }
  }
  return false;
}

/**
 * Returns true if the lightcache can be loaded correctly with this version of eevee.
 **/
bool LightCache::version_check() const
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
bool LightCache::create_reflection_texture(void)
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
bool LightCache::create_irradiance_texture(void)
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
bool LightCache::load_static(void)
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

bool LightCache::load(void)
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


  if (!version_check()) {
    return false;
  }
  switch (type) {
    case LIGHTCACHE_TYPE_STATIC:
      return load_static();
    default:
      return false;
  }
}

/* Return memory footprint in bytes. */
uint LightCache::memsize_get(void) const
{
  uint size = 0;
  if (grid_tx.data) {
    size += MEM_allocN_len(grid_tx.data);
  }
  if (cube_tx.data) {
    size += MEM_allocN_len(cube_tx.data);
    for (int mip = 0; mip < mips_len; mip++) {
      size += MEM_allocN_len(cube_mips[mip].data);
    }
  }
  return size;
}

bool LightCache::can_be_saved(void) const
{
  if (grid_tx.data) {
    if (MEM_allocN_len(grid_tx.data) >= INT_MAX) {
      return false;
    }
  }
  if (cube_tx.data) {
    if (MEM_allocN_len(cube_tx.data) >= INT_MAX) {
      return false;
    }
  }
  return true;
}

int64_t LightCache::irradiance_sample_count(void) const
{
  int64_t total_irr_samples = 0;
  for (const LightGridCache &grid : Span(grid_data, grid_len)) {
    total_irr_samples += grid.resolution[0] * grid.resolution[1] * grid.resolution[2];
  }
  return total_irr_samples;
}

void LightCache::update_info(SceneEEVEE *eevee)
{
  LightCache *lcache = reinterpret_cast<LightCache *>(eevee->light_cache_data);

  if (lcache == nullptr) {
    BLI_strncpy(eevee->light_cache_info,
                TIP_("No light cache in this scene"),
                sizeof(eevee->light_cache_info));
    return;
  }

  if (lcache->version_check() == false) {
    BLI_strncpy(eevee->light_cache_info,
                TIP_("Incompatible Light cache version, please bake again"),
                sizeof(eevee->light_cache_info));
    return;
  }

  if (lcache->cube_tx.tex_size[2] > GPU_max_texture_layers()) {
    BLI_strncpy(eevee->light_cache_info,
                TIP_("Error: Light cache is too big for the GPU to be loaded"),
                sizeof(eevee->light_cache_info));
    return;
  }

  if (lcache->flag & LIGHTCACHE_INVALID) {
    BLI_strncpy(eevee->light_cache_info,
                TIP_("Error: Light cache dimensions not supported by the GPU"),
                sizeof(eevee->light_cache_info));
    return;
  }

  if (lcache->flag & LIGHTCACHE_BAKING) {
    BLI_strncpy(
        eevee->light_cache_info, TIP_("Baking light cache"), sizeof(eevee->light_cache_info));
    return;
  }

  if (lcache->can_be_saved() == false) {
    BLI_strncpy(eevee->light_cache_info,
                TIP_("Error: LightCache is too large and will not be saved to disk"),
                sizeof(eevee->light_cache_info));
    return;
  }

  char formatted_mem[15];
  BLI_str_format_byte_unit(formatted_mem, lcache->memsize_get(), false);

  BLI_snprintf(eevee->light_cache_info,
               sizeof(eevee->light_cache_info),
               TIP_("%d Ref. Cubemaps, %ld Irr. Samples (%s in memory)"),
               lcache->cube_len - 1,
               lcache->irradiance_sample_count(),
               formatted_mem);
}

}  // namespace blender::eevee

/** \} */

/* -------------------------------------------------------------------- */
/** \name Light Bake Context
 * \{ */

/** \} */

/* -------------------------------------------------------------------- */
/** \name Light Bake Job
 * \{ */

/** \} */

/* -------------------------------------------------------------------- */
/** \name C interface
 * \{ */

using namespace blender;

/* Light Bake */
struct wmJob *EEVEE_lightbake_job_create(struct wmWindowManager *wm,
                                         struct wmWindow *win,
                                         struct Main *bmain,
                                         struct ViewLayer *view_layer,
                                         struct Scene *scene,
                                         int delay,
                                         int frame)
{
  return nullptr;
}

void *EEVEE_lightbake_job_data_alloc(struct Main *bmain,
                                     struct ViewLayer *view_layer,
                                     struct Scene *scene,
                                     bool run_as_job,
                                     int frame)
{
  return nullptr;
}

void EEVEE_lightbake_job_data_free(void *custom_data)
{
}

void EEVEE_lightbake_update(void *custom_data)
{
}

void EEVEE_lightbake_job(void *custom_data, short *stop, short *do_update, float *progress)
{
}

void EEVEE_lightcache_free(struct LightCache *lcache_)
{
  eevee::LightCache *lcache = reinterpret_cast<eevee::LightCache *>(lcache_);
  OBJECT_GUARDED_SAFE_DELETE(lcache, eevee::LightCache);
}

void EEVEE_lightcache_info_update(struct SceneEEVEE *eevee)
{
  eevee::LightCache::update_info(eevee);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Read / Write
 * \{ */

static void write_lightcache_texture(BlendWriter *writer, LightCacheTexture *tex)
{
  if (tex->data) {
    size_t data_size = tex->components * tex->tex_size[0] * tex->tex_size[1] * tex->tex_size[2];
    if (tex->data_type == LIGHTCACHETEX_FLOAT) {
      data_size *= sizeof(float);
    }
    else if (tex->data_type == LIGHTCACHETEX_UINT) {
      data_size *= sizeof(uint);
    }

    /* FIXME: We can't save more than what 32bit systems can handle.
     * The solution would be to split the texture but it is too late for 2.90.
     * (see T78529) */
    if (data_size < INT_MAX) {
      BLO_write_raw(writer, data_size, tex->data);
    }
  }
}

void EEVEE_lightcache_blend_write(struct BlendWriter *writer, struct LightCache *cache)
{
  write_lightcache_texture(writer, &cache->grid_tx);
  write_lightcache_texture(writer, &cache->cube_tx);

  if (cache->cube_mips) {
    BLO_write_struct_array(writer, LightCacheTexture, cache->mips_len, cache->cube_mips);
    for (int i = 0; i < cache->mips_len; i++) {
      write_lightcache_texture(writer, &cache->cube_mips[i]);
    }
  }

  BLO_write_struct_array(writer, LightGridCache, cache->grid_len, cache->grid_data);
  BLO_write_struct_array(writer, LightProbeCache, cache->cube_len, cache->cube_data);
}

static void direct_link_lightcache_texture(BlendDataReader *reader, LightCacheTexture *lctex)
{
  lctex->tex = NULL;

  if (lctex->data) {
    BLO_read_data_address(reader, &lctex->data);
    if (lctex->data && BLO_read_requires_endian_switch(reader)) {
      int data_size = lctex->components * lctex->tex_size[0] * lctex->tex_size[1] *
                      lctex->tex_size[2];

      if (lctex->data_type == LIGHTCACHETEX_FLOAT) {
        BLI_endian_switch_float_array((float *)lctex->data, data_size * sizeof(float));
      }
      else if (lctex->data_type == LIGHTCACHETEX_UINT) {
        BLI_endian_switch_uint32_array((uint *)lctex->data, data_size * sizeof(uint));
      }
    }
  }

  if (lctex->data == NULL) {
    zero_v3_int(lctex->tex_size);
  }
}

void EEVEE_lightcache_blend_read_data(struct BlendDataReader *reader, struct LightCache *cache)
{
  cache->flag &= ~LIGHTCACHE_NOT_USABLE;
  direct_link_lightcache_texture(reader, &cache->cube_tx);
  direct_link_lightcache_texture(reader, &cache->grid_tx);

  if (cache->cube_mips) {
    BLO_read_data_address(reader, &cache->cube_mips);
    for (int i = 0; i < cache->mips_len; i++) {
      direct_link_lightcache_texture(reader, &cache->cube_mips[i]);
    }
  }

  BLO_read_data_address(reader, &cache->cube_data);
  BLO_read_data_address(reader, &cache->grid_data);
}

/** \} */
