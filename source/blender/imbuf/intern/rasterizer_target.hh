/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

/** \file
 * \ingroup imbuf
 *
 * Rasterizer drawing target.
 */

#include "BLI_sys_types.h"

namespace blender::imbuf::rasterizer {

/**
 * An abstract implementation of a drawing target. Will make it possible to switch to other render
 * targets then only ImBuf types.
 */
template<typename Inner> class AbstractDrawingTarget {
 public:
  using InnerType = Inner;
  virtual uint64_t get_width() const = 0;
  virtual uint64_t get_height() const = 0;
  virtual float *get_pixel_ptr(uint64_t x, uint64_t y) = 0;
  virtual int64_t get_pixel_stride() const = 0;
  virtual bool has_active_target() const = 0;
  virtual void activate(Inner *instance) = 0;
  virtual void deactivate() = 0;
};

class ImageBufferDrawingTarget : public AbstractDrawingTarget<ImBuf> {
 private:
  ImBuf *image_buffer_ = nullptr;

 public:
  bool has_active_target() const override
  {
    return image_buffer_ != nullptr;
  }

  void activate(ImBuf *image_buffer) override
  {
    image_buffer_ = image_buffer;
  }

  void deactivate() override
  {
    image_buffer_ = nullptr;
  }

  uint64_t get_width() const override
  {
    return image_buffer_->x;
  };

  uint64_t get_height() const override
  {
    return image_buffer_->y;
  }

  float *get_pixel_ptr(uint64_t x, uint64_t y) override
  {
    BLI_assert(has_active_target());
    uint64_t pixel_index = y * image_buffer_->x + x;
    return &image_buffer_->rect_float[pixel_index * 4];
  }
  int64_t get_pixel_stride() const override
  {
    return 4;
  }
};

}  // namespace blender::imbuf::rasterizer
