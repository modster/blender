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
 */

/** \file
 * \ingroup bli
 *
 * \brief High-level API for reading and storing different types of files.
 */

/* TODO define public and internal API. */

#pragma once

#include <optional>

#include "BLI_vector.hh"

namespace blender::ed::filelist {

class AbstractFileEntry;

struct FileListReadParams {
  struct RecursionSettings {
    int max_recursion_level_ = 0;
    bool recurse_into_blends_ = false;
  };

  FileListReadParams(std::string path,
                     std::optional<RecursionSettings> recursion_settings = std::nullopt);

  std::string path_;
  std::optional<RecursionSettings> recursion_settings_;
  /* Hidden files? */
  bool skip_current_and_parent_ = true;
};

class AbstractFileList {
 public:
  AbstractFileList() = default;
  virtual ~AbstractFileList() = default;

  virtual void fetch() = 0;
};

/* Note that each entry may have child entries. */
using FileEntires = blender::Vector<std::unique_ptr<AbstractFileEntry>>;

class FileList final : public AbstractFileList {
  FileListReadParams read_params_;

  FileEntires file_entries_;

 public:
  FileList(const FileListReadParams &read_params);

  void fetch() override;

  StringRef rootPath() const;
  std::string fullPathToFile(const AbstractFileEntry &file) const;
  std::string fullFilePathToFile(const AbstractFileEntry &file) const;
};

}  // namespace blender::ed::filelist
