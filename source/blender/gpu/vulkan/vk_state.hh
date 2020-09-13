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
 */

/** \file
 * \ingroup gpu
 */

#pragma once

#include "MEM_guardedalloc.h"

#include "BLI_utildefines.h"

#include "gpu_state_private.hh"

namespace blender {
namespace gpu {

/**
 * State manager keeping track of the draw state and applying it before drawing.
 * Opengl Implementation.
 **/
class VKStateManager : public StateManager {
 public:
  VKStateManager(){};

  void apply_state(void) override{};
  void force_state(void) override{};

  void issue_barrier(eGPUBarrier barrier_bits) override{};

  void texture_bind(Texture *tex, eGPUSamplerState sampler, int unit) override{};
  void texture_unbind(Texture *tex) override{};
  void texture_unbind_all(void) override{};

  void image_bind(Texture *tex, int unit) override{};
  void image_unbind(Texture *tex) override{};
  void image_unbind_all(void) override{};

  void texture_unpack_row_length_set(uint len) override{};

  MEM_CXX_CLASS_ALLOC_FUNCS("VKStateManager")
};

}  // namespace gpu
}  // namespace blender
