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

#include "BLI_rect.h"

struct ImBuf;

namespace blender::compositor {
class MemoryBuffer;

/* -------------------------------------------------------------------- */
/** \name Elements constants
 * \{ */

constexpr float TRANSPARENT_COLOR[4] = {0.0f, 0.0f, 0.0f, 0.0f};
constexpr float ZERO_VECTOR[3] = {0.0f, 0.0f, 0.0f};

/** \} */

/* -------------------------------------------------------------------- */
/** \name MemoryBuffer
 * \{ */

void copy_buffer_rect(MemoryBuffer *dst, MemoryBuffer *src, const rcti &rect);
void copy_buffer_rect(
    MemoryBuffer *dst, int dst_x, int dst_y, MemoryBuffer *src, const rcti &src_rect);
void copy_buffer_rect(MemoryBuffer *dst,
                      int dst_x,
                      int dst_y,
                      int dst_channel_offset,
                      MemoryBuffer *src,
                      const rcti &src_rect,
                      int src_channel_offset,
                      int elem_size);

void fill_buffer_rect(MemoryBuffer *buf, const rcti &rect, float *fill_elem);
void fill_buffer_rect(
    MemoryBuffer *buf, const rcti &rect, int channel_offset, float *fill_elem, int fill_elem_size);

/** \} */

/* -------------------------------------------------------------------- */
/** \name ImBuf
 * \{ */

void copy_buffer_rect(struct ImBuf *dst, MemoryBuffer *src, const rcti &rect);
void copy_buffer_rect(struct ImBuf *dst,
                      int dst_x,
                      int dst_y,
                      int dst_channel_offset,
                      MemoryBuffer *src,
                      const rcti &src_rect,
                      int src_channel_offset,
                      int elem_size);

void copy_buffer_rect(MemoryBuffer *dst,
                      struct ImBuf *src,
                      const rcti &rect,
                      bool ensure_linear_space = false);
void copy_buffer_rect(MemoryBuffer *dst,
                      int dst_x,
                      int dst_y,
                      int dst_channel_offset,
                      struct ImBuf *src,
                      const rcti &src_rect,
                      int src_channel_offset,
                      int elem_size,
                      bool ensure_linear_space = false);

/** \} */

/* -------------------------------------------------------------------- */
/** \name uchar
 * \{ */

void copy_buffer_rect(uchar *dst, MemoryBuffer *src, const rcti &rect);
void copy_buffer_rect(uchar *dst,
                      int dst_x,
                      int dst_y,
                      int dst_channel_offset,
                      int dst_elem_stride,
                      MemoryBuffer *src,
                      const rcti &src_rect,
                      int src_channel_offset,
                      int elem_size);

void copy_buffer_rect(MemoryBuffer *dst, uchar *src, const rcti &rect);
void copy_buffer_rect(MemoryBuffer *dst,
                      int dst_x,
                      int dst_y,
                      int dst_channel_offset,
                      const uchar *src,
                      const rcti &src_rect,
                      int src_channel_offset,
                      int src_elem_stride,
                      int elem_size);

/** \} */

}  // namespace blender::compositor
