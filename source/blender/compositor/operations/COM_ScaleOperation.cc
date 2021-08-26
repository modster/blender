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

#include "COM_ScaleOperation.h"
#include "COM_ConstantOperation.h"

namespace blender::compositor {

#define USE_FORCE_BILINEAR
/* XXX(campbell): ignore input and use default from old compositor,
 * could become an option like the transform node.
 *
 * NOTE: use bilinear because bicubic makes fuzzy even when not scaling at all (1:1)
 */

BaseScaleOperation::BaseScaleOperation()
{
#ifdef USE_FORCE_BILINEAR
  m_sampler = (int)PixelSampler::Bilinear;
#else
  m_sampler = -1;
#endif
  m_variable_size = false;
}

ScaleOperation::ScaleOperation() : ScaleOperation(DataType::Color)
{
}

ScaleOperation::ScaleOperation(DataType data_type) : BaseScaleOperation()
{
  this->addInputSocket(data_type, ResizeMode::Align);
  this->addInputSocket(DataType::Value);
  this->addInputSocket(DataType::Value);
  this->addOutputSocket(data_type);
  this->set_canvas_input_index(IMAGE_INPUT_INDEX);
  this->m_inputOperation = nullptr;
  this->m_inputXOperation = nullptr;
  this->m_inputYOperation = nullptr;
}

float ScaleOperation::get_constant_scale(const int input_op_idx, const float factor)
{
  const bool is_constant = getInputOperation(input_op_idx)->get_flags().is_constant_operation;
  if (is_constant) {
    return ((ConstantOperation *)getInputOperation(input_op_idx))->get_constant_elem()[0] * factor;
  }

  return 1.0f;
}

float ScaleOperation::get_constant_scale_x(const float width)
{
  return get_constant_scale(X_INPUT_INDEX, get_relative_scale_x_factor(width));
}

float ScaleOperation::get_constant_scale_y(const float height)
{
  return get_constant_scale(Y_INPUT_INDEX, get_relative_scale_y_factor(height));
}

void ScaleOperation::scale_area_inverted(
    rcti &rect, float center_x, float center_y, float scale_x, float scale_y)
{
  rect.xmin = scale_coord_inverted(rect.xmin, center_x, scale_x);
  rect.xmax = scale_coord_inverted(rect.xmax, center_x, scale_x);
  rect.ymin = scale_coord_inverted(rect.ymin, center_y, scale_y);
  rect.ymax = scale_coord_inverted(rect.ymax, center_y, scale_y);
}

void ScaleOperation::scale_area(rcti &rect,
                                float scale_offset_x,
                                float scale_offset_y,
                                float relative_scale_x,
                                float relative_scale_y)
{
  rect.xmin = scale_coord(rect.xmin, scale_offset_x, relative_scale_x);
  rect.xmax = scale_coord(rect.xmax, scale_offset_x, relative_scale_x);
  rect.ymin = scale_coord(rect.ymin, scale_offset_y, relative_scale_y);
  rect.ymax = scale_coord(rect.ymax, scale_offset_y, relative_scale_y);
}

void ScaleOperation::init_data()
{
  canvas_center_x_ = canvas_.xmin + getWidth() / 2.0f;
  canvas_center_y_ = canvas_.ymin + getHeight() / 2.0f;
  scale_offset_x_ = (input_width_scaled_ - getWidth()) / 2.0f;
  scale_offset_y_ = (input_height_scaled_ - getHeight()) / 2.0f;
}

void ScaleOperation::initExecution()
{
  this->m_inputOperation = this->getInputSocketReader(0);
  this->m_inputXOperation = this->getInputSocketReader(1);
  this->m_inputYOperation = this->getInputSocketReader(2);
}

void ScaleOperation::deinitExecution()
{
  this->m_inputOperation = nullptr;
  this->m_inputXOperation = nullptr;
  this->m_inputYOperation = nullptr;
}

void ScaleOperation::get_area_of_interest(const int input_idx,
                                          const rcti &output_area,
                                          rcti &r_input_area)
{
  r_input_area = output_area;
  if (input_idx != 0 || m_variable_size) {
    return;
  }

  NodeOperation *input_img = get_input_operation(IMAGE_INPUT_INDEX);
  const float scale_x = get_constant_scale_x(input_img->getWidth());
  const float scale_y = get_constant_scale_y(input_img->getHeight());
  scale_area(r_input_area, scale_offset_x_, scale_offset_y_, scale_x, scale_y);
}

void ScaleOperation::update_memory_buffer_partial(MemoryBuffer *output,
                                                  const rcti &area,
                                                  Span<MemoryBuffer *> inputs)
{
  const MemoryBuffer *input_img = inputs[IMAGE_INPUT_INDEX];
  MemoryBuffer *input_x = inputs[X_INPUT_INDEX];
  MemoryBuffer *input_y = inputs[Y_INPUT_INDEX];
  const float scale_x_factor = get_relative_scale_x_factor(input_img->getWidth());
  const float scale_y_factor = get_relative_scale_y_factor(input_img->getHeight());
  BuffersIterator<float> it = output->iterate_with({input_x, input_y}, area);
  for (; !it.is_end(); ++it) {
    const float rel_scale_x = *it.in(0) * scale_x_factor;
    const float rel_scale_y = *it.in(1) * scale_y_factor;
    const float scaled_x = scale_coord(it.x, scale_offset_x_, rel_scale_x);
    const float scaled_y = scale_coord(it.y, scale_offset_y_, rel_scale_y);
    input_img->read_elem_sampled(scaled_x, scaled_y, (PixelSampler)m_sampler, it.out);
  }
}

void ScaleOperation::determine_canvas(const rcti &preferred_area, rcti &r_area)
{
  if (execution_model_ == eExecutionModel::Tiled) {
    NodeOperation::determine_canvas(preferred_area, r_area);
    return;
  }

  const bool determined = getInputSocket(0)->determine_canvas(preferred_area, r_area);
  if (determined) {
    /* Constant input canvases need to be determined before getting constants. */
    rcti unused;
    if (get_input_operation(X_INPUT_INDEX)->get_flags().is_constant_operation) {
      getInputSocket(X_INPUT_INDEX)->determine_canvas(r_area, unused);
    }
    if (get_input_operation(Y_INPUT_INDEX)->get_flags().is_constant_operation) {
      getInputSocket(Y_INPUT_INDEX)->determine_canvas(r_area, unused);
    }

    const float width = BLI_rcti_size_x(&r_area);
    const float height = BLI_rcti_size_y(&r_area);
    const float scale_x = get_constant_scale_x(width);
    const float scale_y = get_constant_scale_y(height);

    input_width_scaled_ = width * scale_x;
    input_height_scaled_ = height * scale_y;
    const float max_scale_width = width * MAX_CANVAS_SCALE_FACTOR;
    const float max_scale_height = height * MAX_CANVAS_SCALE_FACTOR;
    r_area.xmax = r_area.xmin +
                  (input_width_scaled_ < max_scale_width ? input_width_scaled_ : max_scale_width);
    r_area.ymax = r_area.ymin + (input_height_scaled_ < max_scale_height ? input_height_scaled_ :
                                                                           max_scale_height);

    /* Determine x/y inputs canvases with scaled canvas as preferred. */
    get_input_operation(X_INPUT_INDEX)->unset_canvas();
    get_input_operation(Y_INPUT_INDEX)->unset_canvas();
    getInputSocket(X_INPUT_INDEX)->determine_canvas(r_area, unused);
    getInputSocket(Y_INPUT_INDEX)->determine_canvas(r_area, unused);
  }
}

ScaleRelativeOperation::ScaleRelativeOperation() : ScaleOperation()
{
}

ScaleRelativeOperation::ScaleRelativeOperation(DataType data_type) : ScaleOperation(data_type)
{
}

void ScaleRelativeOperation::executePixelSampled(float output[4],
                                                 float x,
                                                 float y,
                                                 PixelSampler sampler)
{
  PixelSampler effective_sampler = getEffectiveSampler(sampler);

  float scaleX[4];
  float scaleY[4];

  this->m_inputXOperation->readSampled(scaleX, x, y, effective_sampler);
  this->m_inputYOperation->readSampled(scaleY, x, y, effective_sampler);

  const float scx = scaleX[0];
  const float scy = scaleY[0];

  float nx = this->canvas_center_x_ + (x - this->canvas_center_x_) / scx;
  float ny = this->canvas_center_y_ + (y - this->canvas_center_y_) / scy;
  this->m_inputOperation->readSampled(output, nx, ny, effective_sampler);
}

bool ScaleRelativeOperation::determineDependingAreaOfInterest(rcti *input,
                                                              ReadBufferOperation *readOperation,
                                                              rcti *output)
{
  rcti newInput;
  if (!m_variable_size) {
    float scaleX[4];
    float scaleY[4];

    this->m_inputXOperation->readSampled(scaleX, 0, 0, PixelSampler::Nearest);
    this->m_inputYOperation->readSampled(scaleY, 0, 0, PixelSampler::Nearest);

    const float scx = scaleX[0];
    const float scy = scaleY[0];

    newInput.xmax = this->canvas_center_x_ + (input->xmax - this->canvas_center_x_) / scx + 1;
    newInput.xmin = this->canvas_center_x_ + (input->xmin - this->canvas_center_x_) / scx - 1;
    newInput.ymax = this->canvas_center_y_ + (input->ymax - this->canvas_center_y_) / scy + 1;
    newInput.ymin = this->canvas_center_y_ + (input->ymin - this->canvas_center_y_) / scy - 1;
  }
  else {
    newInput.xmax = this->getWidth();
    newInput.xmin = 0;
    newInput.ymax = this->getHeight();
    newInput.ymin = 0;
  }
  return BaseScaleOperation::determineDependingAreaOfInterest(&newInput, readOperation, output);
}

void ScaleAbsoluteOperation::executePixelSampled(float output[4],
                                                 float x,
                                                 float y,
                                                 PixelSampler sampler)
{
  PixelSampler effective_sampler = getEffectiveSampler(sampler);

  float scaleX[4];
  float scaleY[4];

  this->m_inputXOperation->readSampled(scaleX, x, y, effective_sampler);
  this->m_inputYOperation->readSampled(scaleY, x, y, effective_sampler);

  const float scx = scaleX[0];  // target absolute scale
  const float scy = scaleY[0];  // target absolute scale

  const float width = this->getWidth();
  const float height = this->getHeight();
  // div
  float relativeXScale = scx / width;
  float relativeYScale = scy / height;

  float nx = this->canvas_center_x_ + (x - this->canvas_center_x_) / relativeXScale;
  float ny = this->canvas_center_y_ + (y - this->canvas_center_y_) / relativeYScale;

  this->m_inputOperation->readSampled(output, nx, ny, effective_sampler);
}

bool ScaleAbsoluteOperation::determineDependingAreaOfInterest(rcti *input,
                                                              ReadBufferOperation *readOperation,
                                                              rcti *output)
{
  rcti newInput;
  if (!m_variable_size) {
    float scaleX[4];
    float scaleY[4];

    this->m_inputXOperation->readSampled(scaleX, 0, 0, PixelSampler::Nearest);
    this->m_inputYOperation->readSampled(scaleY, 0, 0, PixelSampler::Nearest);

    const float scx = scaleX[0];
    const float scy = scaleY[0];
    const float width = this->getWidth();
    const float height = this->getHeight();
    // div
    float relateveXScale = scx / width;
    float relateveYScale = scy / height;

    newInput.xmax = this->canvas_center_x_ +
                    (input->xmax - this->canvas_center_x_) / relateveXScale;
    newInput.xmin = this->canvas_center_x_ +
                    (input->xmin - this->canvas_center_x_) / relateveXScale;
    newInput.ymax = this->canvas_center_y_ +
                    (input->ymax - this->canvas_center_y_) / relateveYScale;
    newInput.ymin = this->canvas_center_y_ +
                    (input->ymin - this->canvas_center_y_) / relateveYScale;
  }
  else {
    newInput.xmax = this->getWidth();
    newInput.xmin = 0;
    newInput.ymax = this->getHeight();
    newInput.ymin = 0;
  }
  return ScaleOperation::determineDependingAreaOfInterest(&newInput, readOperation, output);
}

// Absolute fixed size
ScaleFixedSizeOperation::ScaleFixedSizeOperation() : BaseScaleOperation()
{
  this->addInputSocket(DataType::Color, ResizeMode::Align);
  this->addOutputSocket(DataType::Color);
  this->set_canvas_input_index(0);
  this->m_inputOperation = nullptr;
  this->m_is_offset = false;
}

void ScaleFixedSizeOperation::init_data()
{
  const NodeOperation *input_op = getInputOperation(0);
  this->m_relX = input_op->getWidth() / (float)this->m_newWidth;
  this->m_relY = input_op->getHeight() / (float)this->m_newHeight;

  /* *** all the options below are for a fairly special case - camera framing *** */
  if (this->m_offsetX != 0.0f || this->m_offsetY != 0.0f) {
    this->m_is_offset = true;

    if (this->m_newWidth > this->m_newHeight) {
      this->m_offsetX *= this->m_newWidth;
      this->m_offsetY *= this->m_newWidth;
    }
    else {
      this->m_offsetX *= this->m_newHeight;
      this->m_offsetY *= this->m_newHeight;
    }
  }

  if (this->m_is_aspect) {
    /* apply aspect from clip */
    const float w_src = input_op->getWidth();
    const float h_src = input_op->getHeight();

    /* destination aspect is already applied from the camera frame */
    const float w_dst = this->m_newWidth;
    const float h_dst = this->m_newHeight;

    const float asp_src = w_src / h_src;
    const float asp_dst = w_dst / h_dst;

    if (fabsf(asp_src - asp_dst) >= FLT_EPSILON) {
      if ((asp_src > asp_dst) == (this->m_is_crop == true)) {
        /* fit X */
        const float div = asp_src / asp_dst;
        this->m_relX /= div;
        this->m_offsetX += ((w_src - (w_src * div)) / (w_src / w_dst)) / 2.0f;
      }
      else {
        /* fit Y */
        const float div = asp_dst / asp_src;
        this->m_relY /= div;
        this->m_offsetY += ((h_src - (h_src * div)) / (h_src / h_dst)) / 2.0f;
      }

      this->m_is_offset = true;
    }
  }
  /* *** end framing options *** */
}

void ScaleFixedSizeOperation::initExecution()
{
  this->m_inputOperation = this->getInputSocketReader(0);
}

void ScaleFixedSizeOperation::deinitExecution()
{
  this->m_inputOperation = nullptr;
}

void ScaleFixedSizeOperation::executePixelSampled(float output[4],
                                                  float x,
                                                  float y,
                                                  PixelSampler sampler)
{
  PixelSampler effective_sampler = getEffectiveSampler(sampler);

  if (this->m_is_offset) {
    float nx = ((x - this->m_offsetX) * this->m_relX);
    float ny = ((y - this->m_offsetY) * this->m_relY);
    this->m_inputOperation->readSampled(output, nx, ny, effective_sampler);
  }
  else {
    this->m_inputOperation->readSampled(
        output, x * this->m_relX, y * this->m_relY, effective_sampler);
  }
}

bool ScaleFixedSizeOperation::determineDependingAreaOfInterest(rcti *input,
                                                               ReadBufferOperation *readOperation,
                                                               rcti *output)
{
  rcti newInput;

  newInput.xmax = (input->xmax - m_offsetX) * this->m_relX + 1;
  newInput.xmin = (input->xmin - m_offsetX) * this->m_relX;
  newInput.ymax = (input->ymax - m_offsetY) * this->m_relY + 1;
  newInput.ymin = (input->ymin - m_offsetY) * this->m_relY;

  return BaseScaleOperation::determineDependingAreaOfInterest(&newInput, readOperation, output);
}

void ScaleFixedSizeOperation::determine_canvas(const rcti &preferred_area, rcti &r_area)
{
  rcti local_preferred = preferred_area;
  local_preferred.xmax = local_preferred.xmin + m_newWidth;
  local_preferred.ymax = local_preferred.ymin + m_newHeight;
  BaseScaleOperation::determine_canvas(local_preferred, r_area);
  r_area.xmax = r_area.xmin + m_newWidth;
  r_area.ymax = r_area.ymin + m_newHeight;
}

void ScaleFixedSizeOperation::get_area_of_interest(const int input_idx,
                                                   const rcti &output_area,
                                                   rcti &r_input_area)
{
  BLI_assert(input_idx == 0);
  UNUSED_VARS_NDEBUG(input_idx);
  r_input_area.xmax = (output_area.xmax - m_offsetX) * this->m_relX;
  r_input_area.xmin = (output_area.xmin - m_offsetX) * this->m_relX;
  r_input_area.ymax = (output_area.ymax - m_offsetY) * this->m_relY;
  r_input_area.ymin = (output_area.ymin - m_offsetY) * this->m_relY;
  expand_area_for_sampler(r_input_area, (PixelSampler)m_sampler);
}

void ScaleFixedSizeOperation::update_memory_buffer_partial(MemoryBuffer *output,
                                                           const rcti &area,
                                                           Span<MemoryBuffer *> inputs)
{
  const MemoryBuffer *input_img = inputs[0];
  PixelSampler sampler = (PixelSampler)m_sampler;
  BuffersIterator<float> it = output->iterate_with({}, area);
  if (this->m_is_offset) {
    for (; !it.is_end(); ++it) {
      const float nx = (it.x - this->m_offsetX) * this->m_relX;
      const float ny = (it.y - this->m_offsetY) * this->m_relY;
      input_img->read_elem_sampled(nx, ny, sampler, it.out);
    }
  }
  else {
    for (; !it.is_end(); ++it) {
      input_img->read_elem_sampled(it.x * this->m_relX, it.y * this->m_relY, sampler, it.out);
    }
  }
}

}  // namespace blender::compositor
