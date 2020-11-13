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
 * The Original Code is Copyright (C) 2020 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup obj
 */

#pragma once

#include "BLI_utility_mixins.hh"

#include "BLI_vector.hh"

#include "IO_wavefront_obj.h"

namespace blender::io::obj {

/**
 * Steal elements' ownership in a range-based for-loop.
 */
template<typename T> struct StealUniquePtr {
  std::unique_ptr<T> owning;
  StealUniquePtr(std::unique_ptr<T> &owning) : owning(std::move(owning))
  {
  }
  T *operator->()
  {
    return owning.operator->();
  }
  T &operator*()
  {
    return owning.operator*();
  }
};

/**
 * Behaves like `std::unique_ptr<Depsgraph, custom_deleter>`.
 * Needed to free a new Depsgraph created for #DAG_EVAL_RENDER.
 */
class OBJDepsgraph : NonMovable, NonCopyable {
 private:
  Depsgraph *depsgraph_ = nullptr;
  bool needs_free_ = false;

 public:
  OBJDepsgraph(const bContext *C, const eEvaluationMode eval_mode);
  ~OBJDepsgraph();

  Depsgraph *get();
  void update_for_newframe();
};

void exporter_main(bContext *C, const OBJExportParams &export_params);

class OBJMesh;
class OBJCurve;

std::pair<Vector<std::unique_ptr<OBJMesh>>, Vector<std::unique_ptr<OBJCurve>>>
filter_supported_objects(Depsgraph *depsgraph, const OBJExportParams &export_params);

bool append_frame_to_filename(const char *filepath, const int frame, char *r_filepath_with_frames);
}  // namespace blender::io::obj
