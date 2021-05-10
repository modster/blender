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
 * \brief High-level file item to be used with #FileList.
 */

#pragma once

#include <string>

#include "BLI_fileops.h"
#include "BLI_fileops_types.h"
#include "BLI_string_ref.hh"

#include "file_list.hh"

namespace blender::ed::filelist {

using bli_direntry = struct ::direntry;

struct FileAliasInfo {
  std::string redirection_path;
  int file_type = 0; /* eFileSel_File_Types */
  /** Indicate if the alias points to a valid target of if the redirection is broken. */
  bool broken = false;
};

class AbstractFileListEntry {
 protected:
  std::string name_;
  /* TODO should this be in the base really? E.g. IDs/assets won't use this probably. */
  /** Defined in BLI_fileops.h */
  eFileAttributes attributes_;
  BLI_stat_t stat_;

  /** The parent file (if any), used to reconstruct the path to the file-list. */
  const AbstractFileListEntry *parent_;

  AbstractFileListEntry(const bli_direntry &direntry,
                        const eFileAttributes attributes,
                        const AbstractFileListEntry *parent = nullptr);

 public:
  virtual ~AbstractFileListEntry() = default;

  void hide();
  bool isHidden() const;

  /* Read-only getters. */
  StringRef name() const;
  eFileAttributes attributes() const;
  const BLI_stat_t &stat() const;

  /**
   * Get the path relative to the file list root directory. Returned as newly allocated string.
   */
  std::string relative_path() const;
  /**
   * Version of #relative_path() for C-style string buffers.
   */
  void relative_path(char *path_buffer, size_t maxsize) const;
  /**
   * Get the path including the name of the file itself, relative to the file list root directory.
   * Returned as newly allocated string.
   */
  std::string relative_file_path() const;
};

class FileEntry : public AbstractFileListEntry {
  std::optional<std::string> redirect_path_;

 public:
  enum class Type {
    Image,
    Movie,
    PyScript,
    FTFont,
    Sound,
    Text,
    Archive,
    BTX,
    Collada,
    /* TODO FILE_TYPE_OPERATOR? */
    ApplicationBundle,
    Alembic,
    ObjectIO,
    USD,
    VOLUME
  };

  FileEntry(const bli_direntry &direntry,
            const eFileAttributes attributes,
            const AbstractFileListEntry *parent);
  virtual ~FileEntry() = default;

  void setAlias(const FileAliasInfo &alias_info);
  bool isAlias() const;
};

class DirectoryEntry : public AbstractFileListEntry {
  std::optional<std::string> redirect_path_;

 protected:
  FileTree children_;

 public:
  DirectoryEntry(const bli_direntry &direntry,
                 const eFileAttributes attributes,
                 const AbstractFileListEntry *parent);
  virtual ~DirectoryEntry() = default;

  void setAlias(const FileAliasInfo &alias_info);
  bool isAlias() const;

  FileTree &children();
};

class BlendEntry : public DirectoryEntry {
  bool is_backup_file_ = false;

 public:
  BlendEntry(const bli_direntry &direntry,
             const eFileAttributes attributes,
             const AbstractFileListEntry *parent);
  virtual ~BlendEntry() = default;
};

class BlendIDEntry : public AbstractFileListEntry {
 public:
  BlendIDEntry(const bli_direntry &direntry,
               const eFileAttributes attributes,
               const AbstractFileListEntry &parent);
  virtual ~BlendIDEntry() = default;
};

class IDAssetEntry : public BlendIDEntry {
 public:
  IDAssetEntry(const bli_direntry &direntry,
               const eFileAttributes attributes,
               const AbstractFileListEntry &parent);
  virtual ~IDAssetEntry() = default;
};

}  // namespace blender::ed::filelist
