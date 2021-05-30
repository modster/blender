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
 * Copyright 2011, Blender Foundation.
 */

#pragma once

#include "COM_NodeOperation.h"

namespace blender::compositor {

/**
 * Base class for primitive (Color/Vector/Value) constant operations. Constant folding is done
 * prior rendering converting all operations that can be constant into Color/Vector/Value
 * operations.
 */
class ConstantOperation : public NodeOperation {
 public:
  ConstantOperation();
  virtual float *get_constant_elem() = 0;

  /* Intentionally non virtual. Constant operations shouldn't need initialization/deinitialization
   * as they are values set beforehand. */
  void initExecution() override;
  void deinitExecution() override;
};

}  // namespace blender::compositor
