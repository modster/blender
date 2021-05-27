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

#include "COM_buffer_util.h"
#include "COM_MemoryBuffer.h"
#include "IMB_colormanagement.h"
#include "IMB_imbuf_types.h"

namespace blender::compositor {

#define ASSERT_BUFFER_CONTAINS_RECT(buffer, rect) \
  BLI_assert(BLI_rcti_inside_rcti(&(buffer)->get_rect(), &(rect)))

#define ASSERT_BUFFER_CONTAINS_RECT_SIZE(dst_buf, dst_x, dst_y, src_rect) \
  BLI_assert((dst_buf)->get_rect().xmin <= (dst_x)); \
  BLI_assert((dst_buf)->get_rect().ymin <= (dst_y)); \
  BLI_assert((dst_buf)->get_rect().xmax >= (dst_x) + BLI_rcti_size_x(&(src_rect))); \
  BLI_assert((dst_buf)->get_rect().ymax >= (dst_y) + BLI_rcti_size_y(&(src_rect)))

#define ASSERT_VALID_ELEM_SIZE(buf, channel_offset, elem_size) \
  BLI_assert((buf)->get_num_channels() >= (channel_offset) + (elem_size))

static void copy_single_elem(MemoryBuffer *dst,
                             const int dst_channel_offset,
                             MemoryBuffer *src,
                             const int src_channel_offset,
                             const int elem_size)
{
  ASSERT_VALID_ELEM_SIZE(dst, dst_channel_offset, elem_size);
  ASSERT_VALID_ELEM_SIZE(src, src_channel_offset, elem_size);
  BLI_assert(dst->is_a_single_elem());

  float *dst_elem = &dst->get_value(
      dst->get_rect().xmin, dst->get_rect().ymin, dst_channel_offset);
  const float *src_elem = &dst->get_value(
      src->get_rect().xmin, src->get_rect().ymin, src_channel_offset);
  const int elem_bytes = elem_size * sizeof(float);
  memcpy(dst_elem, src_elem, elem_bytes);
}

static void copy_rows(
    MemoryBuffer *dst, const int dst_x, const int dst_y, MemoryBuffer *src, const rcti &src_rect)
{
  ASSERT_BUFFER_CONTAINS_RECT(src, src_rect);
  ASSERT_BUFFER_CONTAINS_RECT_SIZE(dst, dst_x, dst_y, src_rect);
  BLI_assert(dst->get_num_channels() == src->get_num_channels());
  BLI_assert(!dst->is_a_single_elem());
  BLI_assert(!src->is_a_single_elem());

  const int width = BLI_rcti_size_x(&src_rect);
  const int height = BLI_rcti_size_y(&src_rect);
  const int row_bytes = dst->get_num_channels() * width * sizeof(float);
  for (int y = 0; y < height; y++) {
    float *dst_row = dst->get_elem(dst_x, dst_y + y);
    const float *src_row = src->get_elem(src_rect.xmin, src_rect.ymin + y);
    memcpy(dst_row, src_row, row_bytes);
  }
}

static void copy_elems(MemoryBuffer *dst,
                       const int dst_x,
                       const int dst_y,
                       const int dst_channel_offset,
                       MemoryBuffer *src,
                       const rcti &src_rect,
                       const int src_channel_offset,
                       const int elem_size)
{
  ASSERT_BUFFER_CONTAINS_RECT(src, src_rect);
  ASSERT_BUFFER_CONTAINS_RECT_SIZE(dst, dst_x, dst_y, src_rect);
  ASSERT_VALID_ELEM_SIZE(dst, dst_channel_offset, elem_size);
  ASSERT_VALID_ELEM_SIZE(src, src_channel_offset, elem_size);

  const int width = BLI_rcti_size_x(&src_rect);
  const int height = BLI_rcti_size_y(&src_rect);
  const int elem_bytes = elem_size * sizeof(float);
  for (int y = 0; y < height; y++) {
    float *dst_elem = &dst->get_value(dst_x, dst_y + y, dst_channel_offset);
    const float *src_elem = &src->get_value(src_rect.xmin, src_rect.ymin + y, src_channel_offset);
    const float *row_end = dst_elem + width * dst->elem_stride;
    while (dst_elem < row_end) {
      memcpy(dst_elem, src_elem, elem_bytes);
      dst_elem += dst->elem_stride;
      src_elem += src->elem_stride;
    }
  }
}

void copy_buffer_rect(MemoryBuffer *dst, MemoryBuffer *src, const rcti &rect)
{
  copy_buffer_rect(dst, rect.xmin, rect.ymin, src, rect);
}

void copy_buffer_rect(
    MemoryBuffer *dst, const int dst_x, const int dst_y, MemoryBuffer *src, const rcti &src_rect)
{
  BLI_assert(dst->get_num_channels() == src->get_num_channels());
  copy_buffer_rect(dst, dst_x, dst_y, 0, src, src_rect, 0, src->get_num_channels());
}

void copy_buffer_rect(MemoryBuffer *dst,
                      const int dst_x,
                      const int dst_y,
                      const int dst_channel_offset,
                      MemoryBuffer *src,
                      const rcti &src_rect,
                      const int src_channel_offset,
                      const int elem_size)
{
  if (dst->is_a_single_elem()) {
    copy_single_elem(dst, dst_channel_offset, src, src_channel_offset, elem_size);
  }
  else if (!src->is_a_single_elem() && elem_size == src->get_num_channels() &&
           elem_size == dst->get_num_channels()) {
    BLI_assert(dst_channel_offset == 0);
    BLI_assert(src_channel_offset == 0);
    copy_rows(dst, dst_x, dst_y, src, src_rect);
  }
  else {
    copy_elems(
        dst, dst_x, dst_y, dst_channel_offset, src, src_rect, src_channel_offset, elem_size);
  }
}

void fill_buffer_rect(MemoryBuffer *buf, const rcti &rect, float *fill_elem)
{
  fill_buffer_rect(buf, rect, 0, fill_elem, buf->get_num_channels());
}

void fill_buffer_rect(MemoryBuffer *buf,
                      const rcti &rect,
                      const int channel_offset,
                      float *fill_elem,
                      const int fill_elem_size)
{
  MemoryBuffer *single_elem = new MemoryBuffer(
      fill_elem, fill_elem_size, buf->getWidth(), buf->getHeight(), true);
  copy_buffer_rect(
      buf, rect.xmin, rect.ymin, channel_offset, single_elem, rect, 0, fill_elem_size);
}

void copy_buffer_rect(uchar *dst, MemoryBuffer *src, const rcti &rect)
{
  copy_buffer_rect(dst,
                   rect.xmin,
                   rect.ymin,
                   0,
                   src->get_num_channels(),
                   src,
                   rect,
                   0,
                   src->get_num_channels());
}

void copy_buffer_rect(uchar *dst,
                      const int dst_x,
                      const int dst_y,
                      const int dst_channel_offset,
                      const int dst_elem_stride,
                      MemoryBuffer *src,
                      const rcti &src_rect,
                      const int src_channel_offset,
                      const int elem_size)
{
  ASSERT_BUFFER_CONTAINS_RECT(src, src_rect);
  ASSERT_VALID_ELEM_SIZE(src, src_channel_offset, elem_size);

  const int width = BLI_rcti_size_x(&src_rect);
  const int height = BLI_rcti_size_y(&src_rect);
  const int dst_row_stride = width * dst_elem_stride;
  uchar *dst_start = dst + dst_y * dst_row_stride + dst_x * dst_elem_stride + dst_channel_offset;
  for (int y = 0; y < height; y++) {
    const float *src_elem = &src->get_value(src_rect.xmin, src_rect.ymin + y, src_channel_offset);
    uchar *dst_elem = dst_start + y * dst_row_stride;
    const uchar *row_end = dst_elem + width * dst_elem_stride;
    while (dst_elem < row_end) {
      rgba_float_to_uchar(dst_elem, src_elem);
      dst_elem += dst_elem_stride;
      src_elem += src->elem_stride;
    }
  }
}

void copy_buffer_rect(MemoryBuffer *dst, uchar *src, const rcti &rect)
{
  copy_buffer_rect(dst,
                   rect.xmin,
                   rect.ymin,
                   0,
                   src,
                   rect,
                   0,
                   dst->get_num_channels(),
                   dst->get_num_channels());
}

void copy_buffer_rect(MemoryBuffer *dst,
                      const int dst_x,
                      const int dst_y,
                      const int dst_channel_offset,
                      const uchar *src,
                      const rcti &src_rect,
                      const int src_channel_offset,
                      const int src_elem_stride,
                      const int elem_size)
{
  ASSERT_BUFFER_CONTAINS_RECT_SIZE(dst, dst_x, dst_y, src_rect);
  ASSERT_VALID_ELEM_SIZE(dst, dst_channel_offset, elem_size);

  const int width = BLI_rcti_size_x(&src_rect);
  const int height = BLI_rcti_size_y(&src_rect);
  const int src_row_stride = width * src_elem_stride;
  const uchar *src_start = src + src_rect.ymin * src_row_stride + src_channel_offset;
  for (int y = 0; y < height; y++) {
    const uchar *src_elem = src_start + y * src_row_stride;
    float *dst_elem = &dst->get_value(dst_x, dst_y + y, dst_channel_offset);
    const float *row_end = dst_elem + width * dst->elem_stride;
    while (dst_elem < row_end) {
      rgba_uchar_to_float(dst_elem, src_elem);
      dst_elem += dst->elem_stride;
      src_elem += src_elem_stride;
    }
  }
}

void copy_buffer_rect(ImBuf *dst, MemoryBuffer *src, const rcti &rect)
{
  copy_buffer_rect(dst, rect.xmin, rect.ymin, 0, src, rect, 0, src->get_num_channels());
}

void copy_buffer_rect(ImBuf *dst,
                      const int dst_x,
                      const int dst_y,
                      const int dst_channel_offset,
                      MemoryBuffer *src,
                      const rcti &src_rect,
                      const int src_channel_offset,
                      const int elem_size)
{
  if (dst->rect_float) {
    MemoryBuffer *dst_buf = new MemoryBuffer(
        dst->rect_float, dst->channels, dst->x, dst->y, false);
    copy_buffer_rect(
        dst_buf, dst_x, dst_y, dst_channel_offset, src, src_rect, src_channel_offset, elem_size);
  }
  else if (dst->rect) {
    uchar *dst_buf = (uchar *)dst->rect;
    const int dst_elem_stride = dst->channels;
    copy_buffer_rect(dst_buf,
                     dst_x,
                     dst_y,
                     dst_channel_offset,
                     dst_elem_stride,
                     src,
                     src_rect,
                     src_channel_offset,
                     elem_size);
  }
  else {
    BLI_assert(
        !"ImBuf dst can't be written because it has neither float or uchar buffer created.");
  }
}

static void colorspace_to_scene_linear(MemoryBuffer *buf,
                                       const rcti &rect,
                                       ColorSpace *from_colorspace)
{
  const int width = BLI_rcti_size_x(&rect);
  const int height = BLI_rcti_size_y(&rect);
  float *out = buf->get_elem(rect.xmin, rect.ymin);
  /* If rect allows continuous memory do conversion in one step. */
  if (buf->getWidth() == width) {
    IMB_colormanagement_colorspace_to_scene_linear(
        out, width, height, buf->get_num_channels(), from_colorspace, false);
  }
  else {
    for (int y = 0; y < height; y++) {
      IMB_colormanagement_colorspace_to_scene_linear(
          out, width, 1, buf->get_num_channels(), from_colorspace, false);
      out += buf->row_stride;
    }
  }
}

void copy_buffer_rect(MemoryBuffer *dst,
                      ImBuf *src,
                      const rcti &rect,
                      const bool ensure_linear_space)
{
  copy_buffer_rect(
      dst, rect.xmin, rect.ymin, 0, src, rect, 0, dst->get_num_channels(), ensure_linear_space);
}

void copy_buffer_rect(MemoryBuffer *dst,
                      const int dst_x,
                      const int dst_y,
                      const int dst_channel_offset,
                      ImBuf *src,
                      const rcti &src_rect,
                      const int src_channel_offset,
                      const int elem_size,
                      const bool ensure_linear_space)
{
  rcti dst_rect;
  BLI_rcti_init(&dst_rect,
                dst_x,
                dst_x + BLI_rcti_size_x(&src_rect),
                dst_y,
                dst_y + BLI_rcti_size_y(&src_rect));
  if (src->rect_float) {
    MemoryBuffer *src_buf = new MemoryBuffer(
        src->rect_float, src->channels, src->x, src->y, false);
    copy_buffer_rect(
        dst, dst_x, dst_y, dst_channel_offset, src_buf, src_rect, src_channel_offset, elem_size);
  }
  else if (src->rect) {
    const uchar *src_buf = (uchar *)src->rect;
    const int src_elem_stride = src->channels;
    copy_buffer_rect(dst,
                     dst_x,
                     dst_y,
                     dst_channel_offset,
                     src_buf,
                     src_rect,
                     src_channel_offset,
                     src_elem_stride,
                     elem_size);
    if (ensure_linear_space) {
      colorspace_to_scene_linear(dst, dst_rect, src->rect_colorspace);
    }
  }
  else {
    /* Empty ImBuf source. Fill destination with zero/empty values. */
    float *fill_elem = new float[elem_size]{0};
    fill_buffer_rect(dst, dst_rect, dst_channel_offset, fill_elem, elem_size);
    delete[] fill_elem;
  }
}

}  // namespace blender::compositor
