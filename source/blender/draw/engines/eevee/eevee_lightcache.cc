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

#include "DEG_depsgraph_build.h"
#include "DEG_depsgraph_query.h"

#include "BKE_object.h"

#include "DNA_collection_types.h"
#include "DNA_lightprobe_types.h"

#include "PIL_time.h"

#include "eevee_instance.hh"
#include "eevee_lightcache.h"
#include "eevee_private.h"

#include "GPU_capabilities.h"
#include "GPU_context.h"

#include "WM_api.h"
#include "WM_types.h"

#include "BLO_read_write.h"

#include "wm_window.h"

#include <mutex>

extern "C" {
/* TODO should be replace by a more elegant alternative. */
extern void DRW_opengl_context_enable(void);
extern void DRW_opengl_context_disable(void);

extern void DRW_opengl_render_context_enable(void *re_gl_context);
extern void DRW_opengl_render_context_disable(void *re_gl_context);
extern void DRW_gpu_render_context_enable(void *re_gpu_context);
extern void DRW_gpu_render_context_disable(void *re_gpu_context);

extern DrawEngineType draw_engine_eevee_type;
}

/* -------------------------------------------------------------------- */
/** \name Light Cache
 * \{ */

namespace blender::eevee {

LightCache::LightCache(const int cube_len_,
                       const int grid_len_,
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
  cube_len = cube_len_;
  grid_len = grid_len_;

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
  /* True during baking. */
  if (grid_len == 0 || cube_len == 0) {
    return false;
  }
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
    /* TODO(fclem) Move to do_version. */
    for (LightGridCache &grid : MutableSpan<LightGridCache>(grid_data, grid_len)) {
      grid.is_ready = 1;
    }
  }

  if (cube_tx.tex == nullptr) {
    if (create_reflection_texture()) {
      for (int mip = 0; mip <= mips_len; mip++) {
        const void *data = (mip == 0) ? cube_tx.data : cube_mips[mip - 1].data;
        GPU_texture_update_mipmap(cube_tx.tex, mip, GPU_DATA_10_11_11_REV, data);
      }
    }
    /* TODO(fclem) Move to do_version. */
    for (LightProbeCache &cube : MutableSpan<LightProbeCache>(cube_data, cube_len)) {
      cube.is_ready = 1;
    }
  }
  return true;
}

bool LightCache::load(void)
{
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

void LightCache::readback_irradiance(void)
{
  MEM_SAFE_FREE(grid_tx.data);
  grid_tx.data = (char *)GPU_texture_read(grid_tx.tex, GPU_DATA_UBYTE, 0);
  grid_tx.data_type = LIGHTCACHETEX_BYTE;
  grid_tx.components = 4;
}

void LightCache::readback_reflections(void)
{
  MEM_SAFE_FREE(cube_tx.data);
  cube_tx.data = (char *)GPU_texture_read(cube_tx.tex, GPU_DATA_10_11_11_REV, 0);
  cube_tx.data_type = LIGHTCACHETEX_UINT;
  cube_tx.components = 1;

  for (int mip = 0; mip < mips_len; mip++) {
    LightCacheTexture &cube_mip = cube_mips[mip];
    MEM_SAFE_FREE(cube_mip.data);
    GPU_texture_get_mipmap_size(cube_tx.tex, mip + 1, cube_mip.tex_size);

    cube_mip.data = (char *)GPU_texture_read(cube_tx.tex, GPU_DATA_10_11_11_REV, mip + 1);
    cube_mip.data_type = LIGHTCACHETEX_UINT;
    cube_mip.components = 1;
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
  for (const LightGridCache &grid : Span(&grid_data[1], grid_len - 1)) {
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
               TIP_("%d Ref. Cubemaps, %lld Irr. Samples (%s in memory)"),
               lcache->cube_len - 1,
               lcache->irradiance_sample_count(),
               formatted_mem);
}

}  // namespace blender::eevee

/** \} */

/* -------------------------------------------------------------------- */
/** \name Light Bake Job
 * \{ */

namespace blender::eevee {

class LightBake {
 private:
  Depsgraph *depsgraph_;
  ViewLayer *view_layer_ = nullptr;
  ViewLayer *view_layer_input_;
  LightCache *lcache_ = nullptr;
  Scene *scene_;
  struct Main *bmain_;
  /** True if this object owns the gl_context_. */
  bool own_resources_ = true;
  /** If the light-cache was created for baking, it's first owned by the baker. */
  bool own_light_cache_ = false;
  /** Scene frame to bake. */
  int frame_ = 0;
  /** ms. delay the start of the baking to not slowdown interactions (TODO remove) */
  int delay_ = 5;
  /** If running in parallel (in a separate thread), use this context. */
  void *gl_context_ = nullptr;
  GPUContext *gpu_context_ = nullptr;

  /** Instance used for baking. */
  Instance *inst_ = nullptr;
  /** Total for all grids. */
  int irradiance_samples_count_;
  /** Number of grid and cube to bake. Data is inside the lightcache. */
  int grid_len_ = 0;
  int cube_len_ = 0;

  /* Copy of probes data for rendering. We could make a lighter copy if needed. */
  Vector<LightProbe> cubes_probe_;
  Vector<LightProbe> grids_probe_;

  /* To compute progress. */
  int64_t total_ = 0, done_ = 0;
  /* Progress bar ratio to update. Only for async bake. */
  float *progress_ = nullptr;
  /* Signal to stop baking. Only for async bake. */
  short *stop_ = nullptr;
  /* Signal to update scene lightcache. Only for async bake. */
  short *do_update_ = nullptr;

  std::mutex mutex_;

  struct GridRenderData {
    LightBake &bake;
    LightGridCache *grid;
    int64_t sample_index;
    int bounce;

    GridRenderData(LightBake &bake_) : bake(bake_){};

    void render(void)
    {
      SceneEEVEE &sce_eevee = DEG_get_evaluated_scene(bake.depsgraph_)->eevee;
      LightProbe *probe = (grid->probe_index > -1) ? &bake.grids_probe_[grid->probe_index] :
                                                     nullptr;

      /* Swap cache on first grid of each bounce. */
      if (bounce > 0 && grid->offset == 0) {
        bake.inst_->lightprobes.swap_irradiance_cache();
      }

      ivec3 cell_co = grid_cell_index_to_coordinate(sample_index, grid->resolution);
      vec3 position = vec3(grid->corner) + vec3(grid->increment_x) * cell_co.x +
                      vec3(grid->increment_y) * cell_co.y + vec3(grid->increment_z) * cell_co.z;

      bake.inst_->lightprobes.bake(bake.depsgraph_,
                                   LIGHTPROBE_TYPE_GRID,
                                   grid->offset + sample_index,
                                   bounce,
                                   position,
                                   probe,
                                   grid->visibility_range);

      /* TODO incremental LVL update. */
      if (sample_index + 1 == (grid->resolution[0] * grid->resolution[1] * grid->resolution[2])) {
        grid->is_ready = 1;
      }
      /* If it's the last grid of the last bounce, tag lighting as updated. */
      if ((grid->offset + sample_index == bake.irradiance_samples_count_ - 1) &&
          (bounce == sce_eevee.gi_diffuse_bounces - 1)) {
        bake.lcache_->flag &= ~LIGHTCACHE_UPDATE_GRID;
      }
    }

    static void callback(void *UNUSED(ved), void *user_data)
    {
      reinterpret_cast<GridRenderData *>(user_data)->render();
    }
  };

  struct CubemapRenderData {
    LightBake &bake;
    LightProbeCache *cube;
    int64_t cube_index_;

    CubemapRenderData(LightBake &bake_) : bake(bake_){};

    void render(void)
    {
      LightProbe *probe = (cube->probe_index > -1) ? &bake.cubes_probe_[cube->probe_index] :
                                                     nullptr;
      bake.inst_->lightprobes.bake(
          bake.depsgraph_, LIGHTPROBE_TYPE_CUBE, cube_index_, 0, cube->position, probe);
      cube->is_ready = 1;

      /* If it's the last probe, tag lighting as updated. */
      if (cube_index_ == bake.cube_len_ - 1) {
        bake.lcache_->flag &= ~LIGHTCACHE_UPDATE_CUBE;
      }
    }

    static void callback(void *UNUSED(ved), void *user_data)
    {
      reinterpret_cast<CubemapRenderData *>(user_data)->render();
    }
  };

 public:
  /* Interupting an existing bake job and reusing its resources if old_bake is not null.
   * Otherwise just create a new bake context. */
  LightBake(struct Main *bmain,
            struct ViewLayer *view_layer,
            struct Scene *scene,
            bool run_as_job,
            int frame,
            int delay,
            LightBake *old_bake = nullptr)
      : /* Cannot reuse depsgraph for now because we cannot get the update from the
         * main database directly. TODO reuse depsgraph and only update positions. */
        depsgraph_(DEG_graph_new(bmain, scene, view_layer, DAG_EVAL_RENDER)),
        view_layer_input_(view_layer),
        scene_(scene),
        bmain_(bmain),
        frame_(frame),
        delay_(delay)
  {
    if (old_bake && (old_bake->view_layer_input_ == view_layer) && (old_bake->bmain_ == bmain)) {
      {
        /* Steal gl_context. */
        std::lock_guard<std::mutex> lock(old_bake->mutex_);
        old_bake->own_resources_ = false;
        gl_context_ = old_bake->gl_context_;
        old_bake->stop();
      }

      if (gl_context_ == nullptr && !GPU_use_main_context_workaround()) {
        gl_context_ = WM_opengl_context_create();
        wm_window_reset_drawable();
      }
    }
    else {
      if (run_as_job && !GPU_use_main_context_workaround()) {
        gl_context_ = WM_opengl_context_create();
        wm_window_reset_drawable();
      }
    }
    BLI_assert(BLI_thread_is_main());
  }

  ~LightBake()
  {
    /* TODO reuse depsgraph. */
    /* if (own_resources_) { */
    DEG_graph_free(depsgraph_);
    /* } */
  }

  void update_scene_cache(void)
  {
    /* If a new light-cache was created, free the old one and reference the new. */
    if (lcache_ && scene_->eevee.light_cache_data != lcache_) {
      if (scene_->eevee.light_cache_data != NULL) {
        EEVEE_lightcache_free(scene_->eevee.light_cache_data);
      }
      scene_->eevee.light_cache_data = lcache_;
      own_light_cache_ = false;
    }
    lcache_->update_info(&scene_->eevee);
    /* Tag to flush the pointer update to eval scenes. */
    DEG_id_tag_update(&scene_->id, ID_RECALC_COPY_ON_WRITE);
  }

  void do_bake(short *stop, short *do_update, float *progress)
  {
    stop_ = stop;
    do_update_ = do_update;
    progress_ = progress;

    DEG_graph_relations_update(depsgraph_);
    DEG_evaluate_on_framechange(depsgraph_, frame_);

    view_layer_ = DEG_get_evaluated_view_layer(depsgraph_);

    context_enable();
    create_resources();

    /* Resource allocation can fail. Early exit in this case. */
    if (lcache_->flag & LIGHTCACHE_INVALID) {
      lcache_->flag &= ~LIGHTCACHE_BAKING;
      this->stop();
      context_disable();
      delete_resources();
      return;
    }

    context_disable();

    /* HACK: Sleep to delay the first rendering operation
     * that causes a small freeze (caused by VBO generation)
     * because this step is locking at this moment. */
    /* TODO remove this. */
    if (delay_) {
      PIL_sleep_ms(delay_);
    }

    SceneEEVEE &sce_eevee = DEG_get_evaluated_scene(depsgraph_)->eevee;

    /* Render world reflections first. Needed for realtime bake preview. */
    if (lcache_->flag & LIGHTCACHE_UPDATE_CUBE) {
      CubemapRenderData cb_data(*this);
      cb_data.cube = &lcache_->cube_data[0];
      cb_data.cube_index_ = 0;
      lightbake_do_sample(CubemapRenderData::callback, &cb_data);
    }
    /* Render irradiance grids. */
    if (lcache_->flag & LIGHTCACHE_UPDATE_GRID) {
      for (int bounce : IndexRange(sce_eevee.gi_diffuse_bounces)) {
        for (LightGridCache &grid : MutableSpan(lcache_->grid_data, grid_len_)) {
          int64_t grid_sample_len = grid.resolution[0] * grid.resolution[1] * grid.resolution[2];
          for (auto sample_index : IndexRange(grid_sample_len)) {
            GridRenderData cb_data(*this);
            cb_data.grid = &grid;
            cb_data.sample_index = sample_index;
            cb_data.bounce = bounce;
            lightbake_do_sample(GridRenderData::callback, &cb_data);
          }
        }
      }
    }
    /* Render reflections. */
    if (lcache_->flag & LIGHTCACHE_UPDATE_CUBE) {
      for (auto cube_index : IndexRange(1, cube_len_ - 1)) {
        CubemapRenderData cb_data(*this);
        cb_data.cube = &lcache_->cube_data[cube_index];
        cb_data.cube_index_ = cube_index;
        lightbake_do_sample(CubemapRenderData::callback, &cb_data);
      }
    }
    /* Read the resulting lighting data to save it to file/disk. */
    context_enable();
    lcache_->readback_irradiance();
    lcache_->readback_reflections();
    context_disable();

    lcache_->flag |= LIGHTCACHE_BAKED;
    lcache_->flag &= ~LIGHTCACHE_BAKING;

    /* Assume that if lbake->gl_context is NULL
     * we are not running in this in a job, so update
     * the scene light-cache pointer before deleting it. */
    if (gl_context_ == nullptr) {
      BLI_assert(BLI_thread_is_main());
      update_scene_cache();
    }

    delete_resources();

    /* Free GPU smoke textures and the smoke domain list correctly: See also
     * T73921.*/
    /* TODO(fclem) is this still needed? */
    // EEVEE_volumes_free_smoke_textures();

    stop_ = nullptr;
    do_update_ = nullptr;
    progress_ = nullptr;
  }

 private:
  bool lightbake_do_sample(void (*render_callback)(void *ved, void *user_data), void *user_data)
  {
    if (G.is_break == true || *stop_) {
      return false;
    }
    /* TODO: make DRW manager instantiable (and only lock on drawing) */
    context_enable();
    DRW_custom_pipeline(&draw_engine_eevee_type, depsgraph_, render_callback, user_data);
    done_ += 1;
    *progress_ = done_ / (float)total_;
    *do_update_ = 1;
    context_disable();
    return true;
  }

  LightGridCache grid_cache_from_object(Object *ob, int probe_index, int64_t &offset)
  {
    LightProbe *probe = (LightProbe *)ob->data;

    LightGridCache grid;
    copy_v3_v3_int(grid.resolution, &probe->grid_resolution_x);

    /* Save current offset and set it for the next grid. */
    grid.offset = offset;
    offset += grid.resolution[0] * grid.resolution[1] * grid.resolution[2];

    /* Add one for level 0 */
    float fac = 1.0f / max_ff(1e-8f, probe->falloff);
    grid.attenuation_scale = fac / max_ff(1e-8f, probe->distinf);
    grid.attenuation_bias = fac;

    /* Update transforms */
    vec3 half_cell_dim = vec3(1.0f) / vec3(UNPACK3(grid.resolution));
    vec3 cell_dim = half_cell_dim * 2.0f;

    /* Matrix converting world space to cell ranges. */
    invert_m4_m4(grid.mat, ob->obmat);

    float4x4 obmat(ob->obmat);

    /* First cell. */
    vec3 corner = obmat * (half_cell_dim - vec3(1.0f));
    copy_v3_v3(grid.corner, corner);

    /* Opposite neighbor cell. */
    vec3 increment_x = (obmat * vec3(cell_dim.x, 0.0f, 0.0f)) - vec3(obmat.values[3]);
    copy_v3_v3(grid.increment_x, increment_x);
    vec3 increment_y = (obmat * vec3(0.0f, cell_dim.y, 0.0f)) - vec3(obmat.values[3]);
    copy_v3_v3(grid.increment_y, increment_y);
    vec3 increment_z = (obmat * vec3(0.0f, 0.0f, cell_dim.z)) - vec3(obmat.values[3]);
    copy_v3_v3(grid.increment_z, increment_z);

    grid.probe_index = probe_index;
    grid.is_ready = 0;
    /* Update level for progressive update. TODO(fclem) port back. */
    grid.level_bias = 1.0f;

    grid.visibility_bias = 0.05f * probe->vis_bias;
    grid.visibility_bleed = probe->vis_bleedbias;
    grid.visibility_range = 1.0f + sqrtf(max_fff(len_squared_v3(grid.increment_x),
                                                 len_squared_v3(grid.increment_y),
                                                 len_squared_v3(grid.increment_z)));
    return grid;
  }

  LightProbeCache cube_cache_from_object(Object *ob, int probe_index)
  {
    LightProbe *probe = (LightProbe *)ob->data;

    LightProbeCache cube;
    /* Update transforms. */
    copy_v3_v3(cube.position, ob->obmat[3]);

    /* Attenuation. */
    cube.attenuation_type = probe->attenuation_type;
    cube.attenuation_fac = 1.0f / max_ff(1e-8f, probe->falloff);

    unit_m4(cube.attenuationmat);
    scale_m4_fl(cube.attenuationmat, probe->distinf);
    mul_m4_m4m4(cube.attenuationmat, ob->obmat, cube.attenuationmat);
    invert_m4(cube.attenuationmat);

    /* Parallax. */
    unit_m4(cube.parallaxmat);

    if ((probe->flag & LIGHTPROBE_FLAG_CUSTOM_PARALLAX) != 0) {
      cube.parallax_type = probe->parallax_type;
      scale_m4_fl(cube.parallaxmat, probe->distpar);
    }
    else {
      cube.parallax_type = probe->attenuation_type;
      scale_m4_fl(cube.parallaxmat, probe->distinf);
    }

    mul_m4_m4m4(cube.parallaxmat, ob->obmat, cube.parallaxmat);
    invert_m4(cube.parallaxmat);

    cube.probe_index = probe_index;
    cube.is_ready = 0;
    return cube;
  }

  /* Counts and generate lightprobes cache data. Returns irradiance sample total count. */
  int64_t sync_probes(Vector<LightGridCache> &grids_data, Vector<LightProbeCache> &cubes_data)
  {
    /* Start at one to count world sample. */
    int64_t irradiance_samples_count = 1;

    grids_probe_.clear();
    cubes_probe_.clear();

    DEG_OBJECT_ITER_FOR_RENDER_ENGINE_BEGIN (depsgraph_, ob) {
      const int ob_visibility = BKE_object_visibility(ob, DAG_EVAL_RENDER);
      if ((ob_visibility & OB_VISIBLE_SELF) == 0) {
        continue;
      }

      if (ob->type == OB_LIGHTPROBE) {
        LightProbe *prb = (LightProbe *)ob->data;

        if (prb->type == LIGHTPROBE_TYPE_GRID) {
          int probe_index = grids_probe_.append_and_get_index(*prb);
          grids_data.append(grid_cache_from_object(ob, probe_index, irradiance_samples_count));
        }
        else if (prb->type == LIGHTPROBE_TYPE_CUBE) {
          int probe_index = cubes_probe_.append_and_get_index(*prb);
          cubes_data.append(cube_cache_from_object(ob, probe_index));
        }
      }
    }
    DEG_OBJECT_ITER_FOR_RENDER_ENGINE_END;

    auto sort_grids = [](const LightGridCache &a, const LightGridCache &b) {
      return mat4_to_scale(a.mat) < mat4_to_scale(b.mat);
    };
    std::sort(grids_data.begin(), grids_data.end(), sort_grids);

    auto sort_cubes = [](const LightProbeCache &a, const LightProbeCache &b) {
      return mat4_to_scale(a.attenuationmat) < mat4_to_scale(b.attenuationmat);
    };
    std::sort(cubes_data.begin(), cubes_data.end(), sort_cubes);

    LightProbeCache world_cube = {};
    world_cube.probe_index = -1;
    cubes_data.prepend({world_cube});

    LightGridCache world_grid = {};
    world_grid.resolution[0] = world_grid.resolution[1] = world_grid.resolution[2] = 1;
    world_grid.offset = 0;
    world_grid.probe_index = -1;
    grids_data.prepend({world_grid});

    return irradiance_samples_count;
  }

  void create_resources(void)
  {
    Scene *scene_eval = DEG_get_evaluated_scene(depsgraph_);
    SceneEEVEE &sce_eevee = scene_eval->eevee;

    Vector<LightProbeCache> cubes_data;
    Vector<LightGridCache> grids_data;
    int64_t irradiance_samples_count = sync_probes(grids_data, cubes_data);

    grid_len_ = grids_data.size();
    cube_len_ = cubes_data.size();

    ivec3 irradiance_tx_size;
    LightCache::irradiance_cache_size_get(
        sce_eevee.gi_visibility_resolution, irradiance_samples_count, irradiance_tx_size);

    /* Ensure Light Cache is ready to accept new data. If not recreate one.
     * WARNING: All the following must be threadsafe. It's currently protected by the DRW mutex. */
    lcache_ = (LightCache *)sce_eevee.light_cache_data;

    /* TODO validate irradiance and reflection cache independently... */
    if (lcache_ &&
        !lcache_->validate(
            cube_len_, sce_eevee.gi_cubemap_resolution, grid_len_, irradiance_tx_size)) {
      /* Note: this is only the scene eval data. This does not count as ownership.
       * Real owner is original scene which gets the new lightcache in update_scene_cache(). */
      sce_eevee.light_cache_data = lcache_ = nullptr;
    }

    if (lcache_ == nullptr) {
      lcache_ = new LightCache(cube_len_,
                               grid_len_,
                               sce_eevee.gi_cubemap_resolution,
                               sce_eevee.gi_visibility_resolution,
                               irradiance_tx_size);
      own_light_cache_ = true;
      /* Note: this is only the scene eval data. This does not count as ownership. */
      sce_eevee.light_cache_data = lcache_;
    }

    /* Copy gathered data to cache. */
    memcpy(lcache_->cube_data, cubes_data.data(), cube_len_ * sizeof(*lcache_->cube_data));
    memcpy(lcache_->grid_data, grids_data.data(), grid_len_ * sizeof(*lcache_->grid_data));

    lcache_->load();
    lcache_->flag |= LIGHTCACHE_BAKING;

    inst_ = reinterpret_cast<Instance *>(EEVEE_instance_alloc());

    done_ = 0;
    total_ = irradiance_samples_count * sce_eevee.gi_diffuse_bounces + grid_len_;
  }

  void delete_resources(void)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (gl_context_) {
      DRW_opengl_render_context_enable(gl_context_);
      DRW_gpu_render_context_enable(gpu_context_);
    }
    else {
      DRW_opengl_context_enable();
    }
    /* XXX Free the resources contained in the viewlayer data
     * to be able to free the context before deleting the depsgraph.  */
    /* TODO(fclem) This is not necessary for now because we do not store anything in view layers
     * since the start of EEVEE's rewrite. But this might change. */
    // EEVEE_view_layer_data_free(sldata_);

    if (inst_) {
      EEVEE_instance_free(reinterpret_cast<EEVEE_Instance *>(inst_));
    }

    if (gpu_context_) {
      DRW_gpu_render_context_disable(gpu_context_);
      DRW_gpu_render_context_enable(gpu_context_);
      GPU_context_discard(gpu_context_);
    }

    if (gl_context_ && own_resources_) {
      /* Delete the baking context. */
      DRW_opengl_render_context_disable(gl_context_);
      WM_opengl_context_dispose(gl_context_);
      gpu_context_ = nullptr;
      gl_context_ = nullptr;
    }
    else if (gl_context_) {
      DRW_opengl_render_context_disable(gl_context_);
    }
    else {
      DRW_opengl_context_disable();
    }
  }

  /* Stop baking (only if async). Threadsafety is responsibility of the caller. */
  void stop(void)
  {
    if (stop_ != nullptr) {
      *stop_ = 1;
    }
    if (do_update_ != nullptr) {
      *do_update_ = 1;
    }
  }

  void context_enable(void)
  {
    if (GPU_use_main_context_workaround() && !BLI_thread_is_main()) {
      GPU_context_main_lock();
      DRW_opengl_context_enable();
      return;
    }

    if (gl_context_) {
      DRW_opengl_render_context_enable(gl_context_);
      if (gpu_context_ == NULL) {
        gpu_context_ = GPU_context_create(NULL);
      }
      DRW_gpu_render_context_enable(gpu_context_);
    }
    else {
      DRW_opengl_context_enable();
    }
  }

  void context_disable(void)
  {
    if (GPU_use_main_context_workaround() && !BLI_thread_is_main()) {
      DRW_opengl_context_disable();
      GPU_context_main_unlock();
      return;
    }

    if (gl_context_) {
      DRW_gpu_render_context_disable(gpu_context_);
      DRW_opengl_render_context_disable(gl_context_);
    }
    else {
      DRW_opengl_context_disable();
    }
  }

  MEM_CXX_CLASS_ALLOC_FUNCS("EEVEE:LightBake")
};

}  // namespace blender::eevee

/** \} */

/* -------------------------------------------------------------------- */
/** \name C interface
 * \{ */

using namespace blender;

/**
 * Allocate a lightbake object to run async baking.
 * MUST run on the main thread.
 **/
struct wmJob *EEVEE_lightbake_job_create(struct wmWindowManager *wm,
                                         struct wmWindow *win,
                                         struct Main *bmain,
                                         struct ViewLayer *view_layer,
                                         struct Scene *scene,
                                         int delay,
                                         int frame)
{
  /* Only one render job at a time. */
  if (WM_jobs_test(wm, scene, WM_JOB_TYPE_RENDER)) {
    return nullptr;
  }

  wmJob *wm_job = WM_jobs_get(wm,
                              win,
                              scene,
                              "Bake Lighting",
                              WM_JOB_EXCL_RENDER | WM_JOB_PRIORITY | WM_JOB_PROGRESS,
                              WM_JOB_TYPE_LIGHT_BAKE);

  /* If job exists do not recreate context and depsgraph. */
  auto *old_lbake = (eevee::LightBake *)WM_jobs_customdata_get(wm_job);

  auto *lbake = new eevee::LightBake(bmain, view_layer, scene, true, frame, delay, old_lbake);

  WM_jobs_customdata_set(wm_job, lbake, EEVEE_lightbake_job_data_free);
  WM_jobs_timer(wm_job, 0.4, NC_SCENE | NA_EDITED, 0);
  WM_jobs_callbacks(
      wm_job, EEVEE_lightbake_job, nullptr, EEVEE_lightbake_update, EEVEE_lightbake_update);

  G.is_break = false;

  return wm_job;
}

/**
 * Allocate a lightbake object to run blocking baking. MUST run on the main thread.
 **/
void *EEVEE_lightbake_job_data_alloc(struct Main *bmain,
                                     struct ViewLayer *view_layer,
                                     struct Scene *scene,
                                     /* TODO(fclem) remove */
                                     bool UNUSED(run_as_job),
                                     int frame)
{
  return new eevee::LightBake(bmain, view_layer, scene, false, frame, 0);
}

void EEVEE_lightbake_job_data_free(void *custom_data)
{
  delete reinterpret_cast<eevee::LightBake *>(custom_data);
}

/**
 * Update function that swaps the lightcache in the scene by the one being baked if it is
 * already renderable.
 **/
void EEVEE_lightbake_update(void *custom_data)
{
  reinterpret_cast<eevee::LightBake *>(custom_data)->update_scene_cache();
}

void EEVEE_lightbake_job(void *custom_data, short *stop, short *do_update, float *progress)
{
  reinterpret_cast<eevee::LightBake *>(custom_data)->do_bake(stop, do_update, progress);
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
  lctex->tex = nullptr;

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

  if (lctex->data == nullptr) {
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
