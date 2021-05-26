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

#include "BLO_readfile.h"

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

void RecursiveFileListReader::read(FileEntires &dest_file_tree)
{
  current_recursion_level_ = 0;
  readDirectory(dest_file_tree, read_params_.path_, nullptr);
}

void RecursiveFileListReader::readDirectory(FileEntires &dest_file_tree,
                                            const blender::StringRef path,
                                            AbstractFileEntry *parent)
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
    AbstractFileEntry &new_entry = addEntry(dest_file_tree, childpath, current_file, parent);

    DirectoryEntry *dir = dynamic_cast<DirectoryEntry *>(&new_entry);
    /* TODO should we recurse into symlinks? */
    if (dir && shouldContinueRecursion()) {
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
#ifdef WIN32
    /* On Windows don't show ".lnk" extension for valid shortcuts. */
    BLI_path_extension_replace(targetpath, FILE_MAXDIR, "");
#endif

    return_info.redirection_path = targetpath;
    if (BLI_is_dir(targetpath)) {
      return_info.type = FileAliasInfo::Type::Directory;
    }
    else {
      return_info.type = FileAliasInfo::Type::File;
    }
  }
  else {
    /* TODO hide file. */
    return_info.broken = true;
  }

  return return_info;
}

template<typename Type>
static Type &create_entry(FileEntires &dest_file_tree,
                          const bli_direntry &direntry,
                          const eFileAttributes attributes,
                          AbstractFileEntry *parent)
{
  dest_file_tree.append_as(std::make_unique<Type>(direntry, attributes, parent));
  return static_cast<Type &>(*dest_file_tree.last());
}

static bool is_hidden_dot_filename(const AbstractFileEntry &file)
{
  StringRef file_name = file.name();
  if (file_name[0] == '.' && !ELEM(file_name[1], '.', '\0')) {
    return true; /* ignore .file */
  }

  if (!file_name.is_empty() && file_name.back() == '~') {
    return true; /* ignore file~ */
  }

  /* Check the file's parents (relative to the file-list root) if any of them is hidden. */
  for (const AbstractFileEntry *parent = file.parent(); parent != nullptr;
       parent = parent->parent()) {
    if (is_hidden_dot_filename(*parent)) {
      return true;
    }
  }

  return false;
}

static bool should_hide_file(const AbstractFileEntry &file, const FileAliasInfo *alias_info)
{
  if (alias_info && alias_info->broken) {
    return true;
  }

#ifndef WIN32
  /* Set Linux-style dot files hidden too. */
  if (is_hidden_dot_filename(file)) {
    return true;
  }
#endif

  return false;
}

AbstractFileEntry &RecursiveFileListReader::addEntry(FileEntires &dest_file_tree,
                                                     const blender::StringRef path,
                                                     const bli_direntry &direntry,
                                                     AbstractFileEntry *parent)
{
  BLI_assert(!shouldSkipFile(direntry));
  eFileAttributes attributes = BLI_file_attributes(path.data());
  std::optional<FileAliasInfo> alias_info = create_alias_info(direntry, attributes);
  FileAliasInfo *alias_info_ptr = alias_info ? &*alias_info : nullptr;

  AbstractFileEntry *new_entry = nullptr;

  if (isDirectory(direntry.s, path, alias_info_ptr)) {
    DirectoryEntry &dir = create_entry<DirectoryEntry>(
        dest_file_tree, direntry, attributes, parent);
    new_entry = &dir;
  }
  else {
    FileEntry &file = create_entry<FileEntry>(dest_file_tree, direntry, attributes, parent);
    new_entry = &file;
  }

  if (alias_info) {
    new_entry->setAlias(*alias_info);
  }

  if (should_hide_file(*new_entry, alias_info_ptr)) {
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
    return alias_info->type == FileAliasInfo::Type::Directory;
  }

#ifndef __APPLE__
  UNUSED_VARS(path);
#endif

  return false;
}

bool RecursiveFileListReader::isBlend(const StringRef path, const FileAliasInfo *alias_info)
{
  const char *target_path = alias_info ? alias_info->redirection_path.data() : path.data();
  return BLO_has_bfile_extension(target_path);
}

bool RecursiveFileListReader::shouldContinueRecursion() const
{
  return read_params_.recursion_settings_ &&
         (current_recursion_level_ < read_params_.recursion_settings_->max_recursion_level_);
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
