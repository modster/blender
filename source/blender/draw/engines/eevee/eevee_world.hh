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
 * World rendering with material handling. Also take care of lookdev
 * HDRI and default material.
 */

#pragma once

#include "DNA_world_types.h"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name Default World Nodetree
 *
 * In order to support worlds without nodetree we reuse and configure a standalone nodetree that
 * we pass for shader generation. The GPUMaterial is still stored inside the World even if
 * it does not use a nodetree.
 * \{ */

class DefaultWorldNodeTree {
 private:
  bNodeTree *ntree_;
  bNodeSocketValueRGBA *color_socket_;

 public:
  DefaultWorldNodeTree();
  ~DefaultWorldNodeTree();

  bNodeTree *nodetree_get(::World *world);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name World
 *
 * \{ */

class World {
 private:
  Instance &inst_;

  DefaultWorldNodeTree default_tree;

  /* Used to detect if world change. */
  ::World *prev_original_world = nullptr;

 public:
  World(Instance &inst) : inst_(inst){};

  void sync(void);
};

/** \} */

}  // namespace blender::eevee
