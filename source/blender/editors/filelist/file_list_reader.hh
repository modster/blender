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

#pragma once

/** \file
 * \ingroup bli
 */

#include "BLI_string_ref.hh"

namespace blender::ed::filelist {

struct FileListReadParams;
struct FileAliasInfo;
class AbstractFileEntry;

using FileEntires = blender::Vector<std::unique_ptr<AbstractFileEntry>>;
using bli_direntry = struct ::direntry;

/**
 * Read the content of a directory and its sub-directories, up to a number of maximum recursions
 * (or infinitely if requested).
 *
 * Does not recurse into .blend files, that is to be handled separately by #BlendFileListIDReader.
 * This is not just for architectural reasons, but to allow reading multiple .blends in parallel,
 * handled by the file list after #RecursiveFileListReader read the normal files.
 *
 * TODO how exactly are redirects handled? (Needs to be defined and documented here.)
 */
class RecursiveFileListReader {
  const FileListReadParams &read_params_;
  int current_recursion_level_ = 0;

 public:
  RecursiveFileListReader(const FileListReadParams &read_params);
  ~RecursiveFileListReader() = default;

  int64_t peekAndCountFiles();

  /**
   * Reads the file list recursively, with the path and max-recursions defined by #read_params_.
   */
  void read(blender::ed::filelist::FileEntires &dest_file_tree);

 private:
  void readDirectory(FileEntires &dest_file_tree,
                     const blender::StringRef path,
                     AbstractFileEntry *parent);
  bool shouldContinueRecursion() const;
  static AbstractFileEntry &addEntry(FileEntires &dest_file_tree,
                                     const blender::StringRef path,
                                     const blender::ed::filelist::bli_direntry &file,
                                     AbstractFileEntry *parent);

  static bool isDirectory(const BLI_stat_t &stat,
                          const StringRef path,
                          const FileAliasInfo *alias_info);
  static bool isBlend(const StringRef path, const FileAliasInfo *alias_info);
  bool shouldSkipFile(const bli_direntry &) const;
  bool shouldSkipFile(const char *file_name) const;

  friend bool count_files_cb(const char *, const char *, const BLI_stat_t *, void *);
};

/**
 * Read the data-blocks of a .blend into file entries.
 */
class BlendFileListIDReader {
  /* TODO */
};

}  // namespace blender::ed::filelist
