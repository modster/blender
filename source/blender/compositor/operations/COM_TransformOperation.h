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

#pragma once

#include "COM_MultiThreadedOperation.h"

namespace blender::compositor {

class TransformOperation : public MultiThreadedOperation {
 private:
  float scale_center_x_;
  float scale_center_y_;
  float rotate_center_x_;
  float rotate_center_y_;
  float rotate_cosine_;
  float rotate_sine_;
  float translate_x_;
  float translate_y_;
  float constant_scale_;

  /* Set variables. */
  PixelSampler sampler_;
  bool convert_degree_to_rad_;
  float translate_factor_x_;
  float translate_factor_y_;

 public:
  TransformOperation();

  void set_translate_factor_xy(float x, float y)
  {
    translate_factor_x_ = x;
    translate_factor_y_ = y;
  }
  void set_convert_rotate_degree_to_rad(bool value)
  {
    convert_degree_to_rad_ = value;
  }
  void set_sampler(PixelSampler sampler)
  {
    sampler_ = sampler;
  }
  void init_data() override;
  void get_area_of_interest(int input_idx, const rcti &output_area, rcti &r_input_area) override;
  void update_memory_buffer_partial(MemoryBuffer *output,
                                    const rcti &area,
                                    Span<MemoryBuffer *> inputs) override;
};

}  // namespace blender::compositor
