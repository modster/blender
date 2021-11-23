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
 * Copyright 2021, Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * The light module manages light data buffers and light culling system.
 */

#pragma once

#include "BLI_bitmap.h"
#include "BLI_vector.hh"
#include "DNA_light_types.h"

#include "eevee_camera.hh"
#include "eevee_id_map.hh"
#include "eevee_sampling.hh"
#include "eevee_shader.hh"
#include "eevee_shader_shared.hh"
#include "eevee_shadow.hh"
#include "eevee_wrapper.hh"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Light Object
 * \{ */

struct Light : public LightData {
 public:
  bool initialized = false;
  bool used = false;

 public:
  Light()
  {
    shadow_id = LIGHT_NO_SHADOW;
  }

  void sync(ShadowModule &shadows, const Object *ob, float threshold);

  void shadow_discard_safe(ShadowModule &shadows);

  void debug_draw(void);

 private:
  float attenuation_radius_get(const ::Light *la, float light_threshold, float light_power);
  void shape_parameters_set(const ::Light *la, const float scale[3]);
  float shape_power_get(const ::Light *la);
  float point_power_get(const ::Light *la);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name LightModule
 * \{ */

/**
 * The light module manages light data buffers and light culling system.
 */
class LightModule {
  friend ShadowModule;

 public:
  /** Scene lights data. */
  LightDataBuf lights_data;
  /** Shadow data. TODO(fclem): merge with lights_data. */
  ShadowDataBuf shadows_data;
  /** Culling infos. */
  CullingDataBuf culling_data;
  /** Key buffer containing only visible lights indices. */
  CullingKeyBuf culling_key_buf;
  /** LightData buffer used for rendering. Ordered by the culling phase. */
  CullingLightBuf culling_light_buf;
  /** Zbins containing min and max light index for each Z bin. */
  CullingZbinBuf culling_zbin_buf;
  /** Bitmap of lights touching each tiles. Using one layer for each culling batch. */
  CullingTileBuf culling_tile_buf;

 private:
  Instance &inst_;

  /** Map of light objects. This is used to track light deletion. */
  Map<ObjectKey, Light> lights_;

  Vector<Light *> light_refs_;

  /** Follows the principles of Tiled Culling + Z binning from:
   * "Improved Culling for Tiled and Clustered Rendering"
   * by Michal Drobot
   * http://advances.realtimerendering.com/s2017/2017_Sig_Improved_Culling_final.pdf */
  DRWPass *culling_ps_ = nullptr;
  int3 culling_tile_dispatch_size_ = int3(1);
  /* Number of batches of lights that are separately processed. */
  int batch_len_ = 1;

  float light_threshold_;

  /** Debug Culling visualization. */
  DRWPass *debug_draw_ps_ = nullptr;
  GPUTexture *input_depth_tx_ = nullptr;

 public:
  LightModule(Instance &inst) : inst_(inst){};
  ~LightModule(){};

  void begin_sync(void);
  void sync_light(const Object *ob, ObjectHandle &handle);
  void end_sync(void);

  void set_view(const DRWView *view, const ivec2 extent, bool enable_specular = true);

  void shgroup_resources(DRWShadingGroup *grp);

  void debug_end_sync(void);
  void debug_draw(GPUFrameBuffer *view_fb, HiZBuffer &hiz);
};

/** \} */

}  // namespace blender::eevee
