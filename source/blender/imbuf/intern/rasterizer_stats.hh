/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "BLI_sys_types.h"

namespace blender::imbuf::rasterizer {

class AbstractStats {
 public:
  virtual void increase_triangles() = 0;
  virtual void increase_discarded_triangles() = 0;
  virtual void increase_flushes() = 0;
  virtual void increase_rasterlines() = 0;
  virtual void increase_clamped_rasterlines() = 0;
  virtual void increase_discarded_rasterlines() = 0;
  virtual void increase_drawn_fragments(uint64_t fragments_drawn) = 0;
};

class Stats : public AbstractStats {
 public:
  int64_t triangles = 0;
  int64_t discarded_triangles = 0;
  int64_t flushes = 0;
  int64_t rasterlines = 0;
  int64_t clamped_rasterlines = 0;
  int64_t discarded_rasterlines = 0;
  int64_t drawn_fragments = 0;

  void increase_triangles() override
  {
    triangles += 1;
  }

  void increase_discarded_triangles() override
  {
    discarded_triangles += 1;
  }

  void increase_flushes() override
  {
    flushes += 1;
  }

  void increase_rasterlines() override
  {
    rasterlines += 1;
  }

  void increase_clamped_rasterlines() override
  {
    clamped_rasterlines += 1;
  }
  void increase_discarded_rasterlines() override
  {
    discarded_rasterlines += 1;
  }
  void increase_drawn_fragments(uint64_t fragments_drawn) override
  {
    drawn_fragments += fragments_drawn;
  }
};

class NullStats : public AbstractStats {
 public:
  void increase_triangles() override{};
  void increase_discarded_triangles() override{};
  void increase_flushes() override{};
  void increase_rasterlines() override{};
  void increase_clamped_rasterlines() override{};
  void increase_discarded_rasterlines() override{};
  void increase_drawn_fragments(uint64_t UNUSED(fragments_drawn)) override
  {
  }
};

}  // namespace blender::imbuf::rasterizer
