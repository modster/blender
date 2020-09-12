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
 * Copyright 2020, Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 *
 * GPU geometry batch
 * Contains VAOs + VBOs + Shader representing a drawable entity.
 */

#pragma once

#include "MEM_guardedalloc.h"

#include "gpu_batch_private.hh"

#include "vk_index_buffer.hh"
#include "vk_vertex_buffer.hh"

namespace blender {
namespace gpu {

class VKBatch : public Batch {
 public:
  VKBatch(){};
  ~VKBatch(){};

  void draw(int v_first, int v_count, int i_first, int i_count) override{};

  /* Convenience getters. */
  VKIndexBuf *elem_(void) const
  {
    return static_cast<VKIndexBuf *>(unwrap(elem));
  }
  VKVertBuf *verts_(const int index) const
  {
    return static_cast<VKVertBuf *>(unwrap(verts[index]));
  }
  VKVertBuf *inst_(const int index) const
  {
    return static_cast<VKVertBuf *>(unwrap(inst[index]));
  }

  MEM_CXX_CLASS_ALLOC_FUNCS("VKBatch");
};

}  // namespace gpu
}  // namespace blender
