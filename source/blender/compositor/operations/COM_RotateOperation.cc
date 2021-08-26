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

#include "COM_RotateOperation.h"
#include "COM_ConstantOperation.h"

#include "BLI_math.h"

namespace blender::compositor {

RotateOperation::RotateOperation()
{
  this->addInputSocket(DataType::Color, ResizeMode::Align);
  this->addInputSocket(DataType::Value);
  this->addOutputSocket(DataType::Color);
  this->set_canvas_input_index(0);
  this->m_imageSocket = nullptr;
  this->m_degreeSocket = nullptr;
  this->m_doDegree2RadConversion = false;
  this->m_isDegreeSet = false;
  sampler_ = PixelSampler::Nearest;
}

void RotateOperation::get_area_rotation_bounds(const rcti &area,
                                               const float center_x,
                                               const float center_y,
                                               const float sine,
                                               const float cosine,
                                               rcti &r_bounds)
{
  const float dxmin = area.xmin - center_x;
  const float dymin = area.ymin - center_y;
  const float dxmax = area.xmax - center_x;
  const float dymax = area.ymax - center_y;

  const float x1 = center_x + (cosine * dxmin + sine * dymin);
  const float x2 = center_x + (cosine * dxmax + sine * dymin);
  const float x3 = center_x + (cosine * dxmin + sine * dymax);
  const float x4 = center_x + (cosine * dxmax + sine * dymax);
  const float y1 = center_y + (-sine * dxmin + cosine * dymin);
  const float y2 = center_y + (-sine * dxmax + cosine * dymin);
  const float y3 = center_y + (-sine * dxmin + cosine * dymax);
  const float y4 = center_y + (-sine * dxmax + cosine * dymax);
  const float minx = MIN2(x1, MIN2(x2, MIN2(x3, x4)));
  const float maxx = MAX2(x1, MAX2(x2, MAX2(x3, x4)));
  const float miny = MIN2(y1, MIN2(y2, MIN2(y3, y4)));
  const float maxy = MAX2(y1, MAX2(y2, MAX2(y3, y4)));

  r_bounds.xmin = floor(minx);
  r_bounds.xmax = ceil(maxx);
  r_bounds.ymin = floor(miny);
  r_bounds.ymax = ceil(maxy);
}

void RotateOperation::init_data()
{
  if (execution_model_ == eExecutionModel::Tiled) {
    get_rotation_center(get_canvas(), m_centerX, m_centerY);
  }

  NodeOperation *input = get_input_operation(0);
  rotate_offset_x_ = ((int)input->getWidth() - (int)getWidth()) / 2.0f;
  rotate_offset_y_ = ((int)input->getHeight() - (int)getHeight()) / 2.0f;
}

void RotateOperation::initExecution()
{
  this->m_imageSocket = this->getInputSocketReader(0);
  this->m_degreeSocket = this->getInputSocketReader(1);
}

void RotateOperation::deinitExecution()
{
  this->m_imageSocket = nullptr;
  this->m_degreeSocket = nullptr;
}

inline void RotateOperation::ensureDegree()
{
  if (!this->m_isDegreeSet) {
    float degree[4];
    switch (execution_model_) {
      case eExecutionModel::Tiled:
        this->m_degreeSocket->readSampled(degree, 0, 0, PixelSampler::Nearest);
        break;
      case eExecutionModel::FullFrame:
        NodeOperation *degree_op = getInputOperation(1);
        const bool is_constant_degree = degree_op->get_flags().is_constant_operation;
        degree[0] = is_constant_degree ?
                        static_cast<ConstantOperation *>(degree_op)->get_constant_elem()[0] :
                        0.0f;
        break;
    }

    double rad;
    if (this->m_doDegree2RadConversion) {
      rad = DEG2RAD((double)degree[0]);
    }
    else {
      rad = degree[0];
    }
    this->m_cosine = cos(rad);
    this->m_sine = sin(rad);

    this->m_isDegreeSet = true;
  }
}

void RotateOperation::executePixelSampled(float output[4], float x, float y, PixelSampler sampler)
{
  ensureDegree();
  const float dy = y - this->m_centerY;
  const float dx = x - this->m_centerX;
  const float nx = this->m_centerX + (this->m_cosine * dx + this->m_sine * dy);
  const float ny = this->m_centerY + (-this->m_sine * dx + this->m_cosine * dy);
  this->m_imageSocket->readSampled(output, nx, ny, sampler);
}

bool RotateOperation::determineDependingAreaOfInterest(rcti *input,
                                                       ReadBufferOperation *readOperation,
                                                       rcti *output)
{
  ensureDegree();
  rcti newInput;

  const float dxmin = input->xmin - this->m_centerX;
  const float dymin = input->ymin - this->m_centerY;
  const float dxmax = input->xmax - this->m_centerX;
  const float dymax = input->ymax - this->m_centerY;

  const float x1 = this->m_centerX + (this->m_cosine * dxmin + this->m_sine * dymin);
  const float x2 = this->m_centerX + (this->m_cosine * dxmax + this->m_sine * dymin);
  const float x3 = this->m_centerX + (this->m_cosine * dxmin + this->m_sine * dymax);
  const float x4 = this->m_centerX + (this->m_cosine * dxmax + this->m_sine * dymax);
  const float y1 = this->m_centerY + (-this->m_sine * dxmin + this->m_cosine * dymin);
  const float y2 = this->m_centerY + (-this->m_sine * dxmax + this->m_cosine * dymin);
  const float y3 = this->m_centerY + (-this->m_sine * dxmin + this->m_cosine * dymax);
  const float y4 = this->m_centerY + (-this->m_sine * dxmax + this->m_cosine * dymax);
  const float minx = MIN2(x1, MIN2(x2, MIN2(x3, x4)));
  const float maxx = MAX2(x1, MAX2(x2, MAX2(x3, x4)));
  const float miny = MIN2(y1, MIN2(y2, MIN2(y3, y4)));
  const float maxy = MAX2(y1, MAX2(y2, MAX2(y3, y4)));

  newInput.xmax = ceil(maxx) + 1;
  newInput.xmin = floor(minx) - 1;
  newInput.ymax = ceil(maxy) + 1;
  newInput.ymin = floor(miny) - 1;

  return NodeOperation::determineDependingAreaOfInterest(&newInput, readOperation, output);
}

void RotateOperation::determine_canvas(const rcti &preferred_area, rcti &r_area)
{
  if (execution_model_ == eExecutionModel::Tiled) {
    NodeOperation::determine_canvas(preferred_area, r_area);
    return;
  }

  const bool determined = getInputSocket(0)->determine_canvas(preferred_area, r_area);
  if (determined) {
    /* Degree input canvas need to be determined before getting its constant. */
    rcti unused;
    if (get_input_operation(1)->get_flags().is_constant_operation) {
      getInputSocket(1)->determine_canvas(r_area, unused);
    }

    ensureDegree();

    get_rotation_center(r_area, m_centerX, m_centerY);

    rcti rot_bounds;
    get_area_rotation_bounds(r_area, m_centerX, m_centerY, m_sine, m_cosine, rot_bounds);
    r_area.xmax = r_area.xmin + BLI_rcti_size_x(&rot_bounds);
    r_area.ymax = r_area.ymin + BLI_rcti_size_y(&rot_bounds);

    /* Determine degree input canvas with scaled canvas as preferred. */
    get_input_operation(1)->unset_canvas();
    getInputSocket(1)->determine_canvas(r_area, unused);
  }
}

void RotateOperation::get_area_of_interest(const int input_idx,
                                           const rcti &output_area,
                                           rcti &r_input_area)
{
  if (input_idx == 1) {
    /* Degrees input is always used as constant. */
    r_input_area = COM_CONSTANT_INPUT_AREA_OF_INTEREST;
    return;
  }

  ensureDegree();

  float center_x;
  float center_y;
  get_rotation_center(get_input_operation(0)->get_canvas(), center_x, center_y);

  r_input_area = output_area;
  BLI_rcti_translate(&r_input_area, rotate_offset_x_, rotate_offset_y_);
  get_area_rotation_bounds(r_input_area, center_x, center_y, m_sine, m_cosine, r_input_area);
  expand_area_for_sampler(r_input_area, sampler_);
}

void RotateOperation::update_memory_buffer_partial(MemoryBuffer *output,
                                                   const rcti &area,
                                                   Span<MemoryBuffer *> inputs)
{
  const MemoryBuffer *input_img = inputs[0];
  float center_x;
  float center_y;
  get_rotation_center(input_img->get_rect(), center_x, center_y);
  for (BuffersIterator<float> it = output->iterate_with({}, area); !it.is_end(); ++it) {
    float x = rotate_offset_x_ + it.x;
    float y = rotate_offset_y_ + it.y;
    rotate_coords(x, y, center_x, center_y, m_sine, m_cosine);
    input_img->read_elem_sampled(x, y, sampler_, it.out);
  }
}

}  // namespace blender::compositor
