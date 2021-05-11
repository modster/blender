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

#include "BLI_fileops.h"
#include "BLI_fileops_types.h"
#include "BLI_path_util.h"

#include "DNA_space_types.h"

#include "ED_fileselect.h"

#include "file_list.hh"
#include "file_list_entry.hh"
#include "file_list_reader.hh"

namespace blender::ed::filelist {

RecursiveFileListReader::RecursiveFileListReader(const FileListReadParams &read_params)
    : read_params_(read_params)
{
}

void RecursiveFileListReader::read(FileTree &dest_file_tree)
{
  current_recursion_level_ = 0;
  readDirectory(dest_file_tree, read_params_.path_, nullptr);
}

void RecursiveFileListReader::readDirectory(FileTree &dest_file_tree,
                                            const blender::StringRef path,
                                            AbstractFileListEntry *parent)
{
  bli_direntry *files;
  uint totfile = BLI_filelist_dir_contents(path.data(), &files);

  for (int i = 0; i < totfile; i++) {
    bli_direntry &current_file = files[i];

    if (shouldSkipFile(current_file)) {
      continue;
    }

    char childpath[FILE_MAX];
    BLI_path_join(childpath, sizeof(childpath), path.data(), current_file.relname, nullptr);
    AbstractFileListEntry &new_entry = addEntry(dest_file_tree, childpath, current_file, parent);

    /* TODO should we recurse into symlinks? */
    if (auto *dir = dynamic_cast<DirectoryEntry *>(&new_entry);
        dir && shouldContinueRecursion() && ShouldRecurseIntoFile(new_entry)) {
      current_recursion_level_++;
      readDirectory(dir->children(), path, dir);
      current_recursion_level_--;
    }
  }

  BLI_filelist_free(files, totfile);
}

std::optional<FileAliasInfo> create_alias_info(const bli_direntry &direntry,
                                               const eFileAttributes attributes)
{

  if ((attributes & FILE_ATTR_ALIAS) == 0) {
    return std::nullopt;
  }

  FileAliasInfo return_info;

  char targetpath[FILE_MAXDIR];
  if (BLI_file_alias_target(direntry.path, targetpath)) {
    return_info.redirection_path = targetpath;
    if (BLI_is_dir(targetpath)) {
      return_info.file_type = FILE_TYPE_DIR;
    }
    else {
      return_info.file_type = ED_path_extension_type(targetpath);
    }
  }
  else {
    /* TODO hide file. */
    return_info.broken = true;
  }

  return return_info;
}

static DirectoryEntry &create_directory(FileTree &dest_file_tree,
                                        const bli_direntry &direntry,
                                        const eFileAttributes attributes,
                                        AbstractFileListEntry *parent)
{
  dest_file_tree.append_as(std::make_unique<DirectoryEntry>(direntry, attributes, parent));
  return static_cast<DirectoryEntry &>(*dest_file_tree.last());
}

static FileEntry &create_file_entry(FileTree &dest_file_tree,
                                    const bli_direntry &direntry,
                                    const eFileAttributes attributes,
                                    AbstractFileListEntry *parent)
{
  dest_file_tree.append_as(std::make_unique<FileEntry>(direntry, attributes, parent));
  return static_cast<FileEntry &>(*dest_file_tree.last());
}

AbstractFileListEntry &RecursiveFileListReader::addEntry(FileTree &dest_file_tree,
                                                         const blender::StringRef path,
                                                         const bli_direntry &direntry,
                                                         AbstractFileListEntry *parent)
{
  BLI_assert(!shouldSkipFile(direntry));
  eFileAttributes attributes = BLI_file_attributes(path.data());
  std::optional<FileAliasInfo> alias_info = create_alias_info(direntry, attributes);

  AbstractFileListEntry *new_entry = nullptr;

  if (isDirectory(direntry.s, path, alias_info ? &*alias_info : nullptr)) {
    DirectoryEntry &dir = create_directory(dest_file_tree, direntry, attributes, parent);
    new_entry = &dir;
  }
  else {
    FileEntry &file = create_file_entry(dest_file_tree, direntry, attributes, parent);
    new_entry = &file;
  }

  if (alias_info && alias_info->broken) {
    new_entry->hide();
  }

  return *new_entry;
}

bool RecursiveFileListReader::isDirectory(const BLI_stat_t &stat,
                                          const StringRef path,
                                          const FileAliasInfo *alias_info)
{
  if (S_ISDIR(stat.st_mode)
#ifdef __APPLE__
      && !(ED_path_extension_type(path.data()) & FILE_TYPE_APPLICATIONBUNDLE)
#endif
  ) {
    return true;
  }

  if (alias_info) {
    return alias_info->file_type == FILE_TYPE_DIR;
  }

#ifndef __APPLE__
  UNUSED_VARS(path);
#endif

  return false;
}

bool RecursiveFileListReader::shouldContinueRecursion() const
{
  return !read_params_.max_recursion_level_ ||
         (current_recursion_level_ < read_params_.max_recursion_level_);
}

bool RecursiveFileListReader::ShouldRecurseIntoFile(const AbstractFileListEntry &entry)
{
  /* TODO "recurse into blends" option isn't implemented at all. */
  const bool recurse_into_blends = false;
  /* Only recurse into blends if requested by the file-list type. */
  if (dynamic_cast<const BlendEntry *>(&entry)) {
    return recurse_into_blends;
  }

  return true;
}

bool RecursiveFileListReader::shouldSkipFile(const bli_direntry &file) const
{
  return shouldSkipFile(file.relname);
}
bool RecursiveFileListReader::shouldSkipFile(const char *file_name) const
{
  return read_params_.skip_current_and_parent_ && FILENAME_IS_CURRPAR(file_name);
}

struct CountFileData {
  int64_t files_counter;
  const FileListReadParams &read_params;
  RecursiveFileListReader &reader;
};

bool count_files_cb(const char *filepath,
                    const char *file_name,
                    const BLI_stat_t *stat,
                    void *customdata)
{
  CountFileData *count_data = static_cast<CountFileData *>(customdata);
  if (count_data->reader.shouldSkipFile(file_name)) {
    return true;
  }

  count_data->files_counter++;

  if (RecursiveFileListReader::isDirectory(*stat, filepath, nullptr) &&
      count_data->reader.shouldContinueRecursion()) {
    count_data->reader.current_recursion_level_++;
    BLI_filelist_dir_contents_iterate_peek(filepath, count_files_cb, count_data);
    count_data->reader.current_recursion_level_--;
  }

  return true;
}

int64_t RecursiveFileListReader::peekAndCountFiles()
{
  CountFileData count_data = {0, read_params_, *this};
  current_recursion_level_ = 0;
  /* TODO this recursion could cause two issues: too many nested calls to `opendir()` and stack
   * overflow due to many file-path buffers. */
  BLI_filelist_dir_contents_iterate_peek(read_params_.path_.data(), count_files_cb, &count_data);

  return count_data.files_counter;
}

}  // namespace blender::ed::filelist
