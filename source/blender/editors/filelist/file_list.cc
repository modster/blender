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

#include <filesystem>
#include <iostream>
#include <string>

#include "BLI_fileops.h"
#include "BLI_fileops_types.h"
#include "BLI_path_util.h"
#include "BLI_string_ref.hh"

#include "file_list.hh"
#include "file_list_entry.hh"
#include "file_list_reader.hh"

namespace blender::ed::filelist {

/**
 * \param max_recursion_level: If not set, recursion is unlimited.
 */
FileListReadParams::FileListReadParams(std::string path,
                                       std::optional<RecursionSettings> recursion_settings)
    : path_(path), recursion_settings_(recursion_settings)
{
}

FileList::FileList(const FileListReadParams &read_params) : read_params_(read_params)
{
}

void print_dir(FileEntires &file_list)
{
  if (file_list.is_empty()) {
    return;
  }
  for (const auto &file : file_list) {
    std::cout << file->relative_file_path() << std::endl;
    if (DirectoryEntry *dir = dynamic_cast<DirectoryEntry *>(file.get())) {
      print_dir(dir->children());
    }
  }
  std::cout << std::endl;
}

void FileList::fetch()
{
  std::cout << "fetch() " << read_params_.path_ << std::endl;

  RecursiveFileListReader reader(read_params_);

  int64_t tot_files = reader.peekAndCountFiles();
  std::cout << tot_files << std::endl;

  reader.read(file_entries_);

  print_dir(file_entries_);
}

StringRef FileList::rootPath() const
{
  return read_params_.path_;
}

/**
 * The full file-path to \a file, excluding the file name.
 */
std::string FileList::fullPathToFile(const AbstractFileEntry &file) const
{
  StringRef root = rootPath();
  std::string rel_file_path = file.relative_path();

  char path[PATH_MAX];
  BLI_path_join(path, sizeof(path), root.data(), rel_file_path.data(), NULL);

  return path;
}

/**
 * The full file-path to \a file, including the file name.
 */
std::string FileList::fullFilePathToFile(const AbstractFileEntry &file) const
{
  StringRef root = rootPath();
  std::string rel_file_path = file.relative_file_path();

  char path[PATH_MAX];
  BLI_path_join(path, sizeof(path), root.data(), rel_file_path.data(), NULL);

  return path;
}

}  // namespace blender::ed::filelist
