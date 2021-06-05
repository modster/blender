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

#include "BKE_studiolight.h"
#include "DNA_world_types.h"

#include "DRW_render.h"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Lookdev World Nodetree
 *
 * \{ */

class LookDevWorldNodeTree {
 private:
  bNodeTree *ntree_;
  bNodeSocketValueFloat *strength_socket_;

 public:
  LookDevWorldNodeTree();
  ~LookDevWorldNodeTree();

  bNodeTree *nodetree_get(float strength);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Look Dev
 *
 * \{ */

class LookDev {
 private:
  Instance &inst_;
  /** Nodetree used to render the world reflection cubemap and irradiance. */
  LookDevWorldNodeTree world_tree;
  /** Compiled gpu material for the nodetree. Owned. */
  ListBase material = {nullptr, nullptr};
  /** Choosen studio light. */
  StudioLight *studiolight_ = nullptr;
  int studiolight_index_ = -1;
  /** Draw pass to draw the viewport background. */
  DRWPass *background_ps_ = nullptr;
  /** Parameters. */
  float instensity_ = -1.0f;
  float blur_ = -1.0f;
  float opacity_ = -1.0f;
  float rotation_ = -9999.0f;
  bool view_rotation_ = false;

  /** Overlay (reference spheres). */
  DRWPass *overlay_ps_ = nullptr;
  /** View based on main view with orthographic projection. Without this, shading is incorrect. */
  DRWView *view_ = nullptr;
  /** Selected LOD of the sphere mesh. */
  eDRWLevelOfDetail sphere_lod_;
  /** Screen space radius in pixels. */
  int sphere_size_ = 0;
  /** Lower right corner of the area where we can start drawing. */
  ivec2 anchor_;

 public:
  LookDev(Instance &inst) : inst_(inst){};
  ~LookDev()
  {
    GPU_material_free(&material);
  };

  void init(const ivec2 &output_res);

  void sync_background(void);
  bool sync_world(void);
  void sync_overlay(void);

  bool render_background(void);
  void render_overlay(GPUFrameBuffer *view_fb);

  void rotation_get(mat4 r_mat);

 private:
  bool do_overlay(void);
};

/** \} */

}  // namespace blender::eevee
