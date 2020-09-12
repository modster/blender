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
 * The Original Code is Copyright (C) 2020 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 *
 * Implementation of Multi Draw Indirect using OpenGL.
 * Fallback if the needed extensions are not supported.
 */

#pragma once

#include "MEM_guardedalloc.h"

#include "BLI_sys_types.h"

#include "GPU_batch.h"

#include "gpu_drawlist_private.hh"

#include "vk_context.hh"

namespace blender {
namespace gpu {

/**
 * Implementation of Multi Draw Indirect using OpenGL.
 **/
class VKDrawList : public DrawList {
 public:
  VKDrawList(int length){};
  ~VKDrawList(){};

  void append(GPUBatch *batch, int i_first, int i_count) override{};
  void submit(void) override{};

  MEM_CXX_CLASS_ALLOC_FUNCS("VKDrawList");
};

}  // namespace gpu
}  // namespace blender
