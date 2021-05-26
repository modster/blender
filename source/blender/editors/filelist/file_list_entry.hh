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

/* Needed for enums. */
#include "DNA_space_types.h"

#include "file_list.hh"

struct Main;

namespace blender::ed::filelist {

using bli_direntry = struct ::direntry;

struct FileAliasInfo {
  enum class Type { Directory, File } type;
  std::string redirection_path;
  /** Indicate if the alias points to a valid target of if the redirection is broken. */
  bool broken = false;
};

class AbstractFileEntry {
 protected:
  std::string name_;
  /* TODO should this be in the base really? E.g. IDs/assets won't use this probably. */
  /** Defined in BLI_fileops.h */
  eFileAttributes attributes_;
  BLI_stat_t stat_;

  /** The parent file (if any), used to reconstruct the path to the file-list. */
  const AbstractFileEntry *parent_;

 public:
  virtual ~AbstractFileEntry() = default;

  virtual void setAlias(const FileAliasInfo &alias_info);

  void hide();
  bool isHidden() const;

  /* Read-only getters. */
  const AbstractFileEntry *parent() const;
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

 protected:
  AbstractFileEntry(const bli_direntry &direntry,
                    const eFileAttributes attributes,
                    const AbstractFileEntry *parent = nullptr);
};

class FileEntry : public AbstractFileEntry {
 public:
  enum class Type {
    Blend,
    BlendBackup,
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
    Volume,
  };

 private:
  std::optional<std::string> redirect_path_;
  Type type_;

 public:
  FileEntry(const bli_direntry &direntry,
            const eFileAttributes attributes,
            const AbstractFileEntry *parent);
  virtual ~FileEntry() = default;

  static Type filesel_type_to_file_entry_type(eFileSel_File_Types type);

  void setAlias(const FileAliasInfo &alias_info);
  bool isAlias() const;
};

class DirectoryEntry : public AbstractFileEntry {
  std::optional<std::string> redirect_path_;

 protected:
  FileEntires children_;

 public:
  DirectoryEntry(const bli_direntry &direntry,
                 const eFileAttributes attributes,
                 const AbstractFileEntry *parent);
  virtual ~DirectoryEntry() = default;

  virtual bool canBeEntered(const Main &current_main, const FileList &file_list) const;

  void setAlias(const FileAliasInfo &alias_info);
  bool isAlias() const;

  FileEntires &children();
};

class BlendEntry : public DirectoryEntry {
  bool is_backup_file_ = false;

 public:
  BlendEntry(const bli_direntry &direntry,
             const eFileAttributes attributes,
             const AbstractFileEntry *parent);
  virtual ~BlendEntry() = default;

  bool canBeEntered(const Main &current_main, const FileList &file_list) const override;
};

class BlendIDEntry : public AbstractFileEntry {
 public:
  BlendIDEntry(const bli_direntry &direntry,
               const eFileAttributes attributes,
               const AbstractFileEntry &parent);
  virtual ~BlendIDEntry() = default;
};

class IDAssetEntry : public BlendIDEntry {
 public:
  IDAssetEntry(const bli_direntry &direntry,
               const eFileAttributes attributes,
               const AbstractFileEntry &parent);
  virtual ~IDAssetEntry() = default;
};

}  // namespace blender::ed::filelist
