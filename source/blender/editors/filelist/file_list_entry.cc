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

#include "BKE_main.h"

#include "BLI_fileops.h"
#include "BLI_path_util.h"

#include "DNA_space_types.h"

#include "ED_fileselect.h"

#include "file_list_entry.hh"

namespace blender::ed::filelist {

AbstractFileEntry::AbstractFileEntry(const bli_direntry &direntry,
                                     const eFileAttributes attributes,
                                     const AbstractFileEntry *parent)
    : name_(direntry.relname),
      /* Could get the attributes via `BLI_file_attributes(direntry.path)` here, but it's needed
         earlier already and it's not too cheap (bunch of file system queries). */
      attributes_(attributes),
      stat_(direntry.s),
      parent_(parent)
{
  BLI_assert(name_.back() != SEP);
}

void AbstractFileEntry::setAlias(const FileAliasInfo &)
{
}

void AbstractFileEntry::hide()
{
  attributes_ |= FILE_ATTR_HIDDEN;
}
bool AbstractFileEntry::isHidden() const
{
  return attributes_ & FILE_ATTR_HIDDEN;
}

const AbstractFileEntry *AbstractFileEntry::parent() const
{
  return parent_;
}

StringRef AbstractFileEntry::name() const
{
  return name_;
}

eFileAttributes AbstractFileEntry::attributes() const
{
  return attributes_;
}

const BLI_stat_t &AbstractFileEntry::stat() const
{
  return stat_;
}

void AbstractFileEntry::relative_path(char *path, size_t maxsize) const
{
  BLI_assert(maxsize > 0);
  path[0] = 0;

  Vector<const char *, 10> path_vec;
  for (const AbstractFileEntry *parent = parent_; parent != nullptr; parent = parent->parent_) {
    path_vec.append(parent->name_.data());
  }

  for (const char *entry : path_vec) {
    BLI_path_append(path, maxsize, entry);
  }
}

std::string AbstractFileEntry::relative_path() const
{
  char path[PATH_MAX];
  relative_path(path, sizeof(path));
  return path;
}

std::string AbstractFileEntry::relative_file_path() const
{
  char path[PATH_MAX];
  relative_path(path, sizeof(path));
  BLI_path_append(path, sizeof(path), name_.data());
  return path;
}

FileEntry::FileEntry(const bli_direntry &direntry,
                     const eFileAttributes attributes,
                     const AbstractFileEntry *parent)
    : AbstractFileEntry(direntry, attributes, parent)
{
  eFileSel_File_Types filesel_type = static_cast<eFileSel_File_Types>(
      ED_path_extension_type(isAlias() ? redirect_path_->data() : name_.data()));
  type_ = filesel_type_to_file_entry_type(filesel_type);
}

FileEntry::Type FileEntry::filesel_type_to_file_entry_type(eFileSel_File_Types type)
{
  switch (type) {
    case FILE_TYPE_BLENDER:
      return Type::Blend;
    case FILE_TYPE_BLENDER_BACKUP:
      return Type::BlendBackup;
    case FILE_TYPE_IMAGE:
      return Type::Image;
    case FILE_TYPE_MOVIE:
      return Type::Movie;
    case FILE_TYPE_PYSCRIPT:
      return Type::PyScript;
    case FILE_TYPE_FTFONT:
      return Type::FTFont;
    case FILE_TYPE_SOUND:
      return Type::Sound;
    case FILE_TYPE_TEXT:
      return Type::Text;
    case FILE_TYPE_ARCHIVE:
      return Type::Archive;
    case FILE_TYPE_BTX:
      return Type::BTX;
    case FILE_TYPE_COLLADA:
      return Type::Collada;
    case FILE_TYPE_OPERATOR:
      /* TODO */
      throw std::exception();
    case FILE_TYPE_APPLICATIONBUNDLE:
      return Type::ApplicationBundle;
    case FILE_TYPE_ALEMBIC:
      return Type::Alembic;
    case FILE_TYPE_OBJECT_IO:
      return Type::ObjectIO;
    case FILE_TYPE_USD:
      return Type::USD;
    case FILE_TYPE_VOLUME:
      return Type::Volume;
      /* These types shouldn't use #FileEntry, they have dedicated classes. */
    case FILE_TYPE_FOLDER:
    case FILE_TYPE_ASSET:
    case FILE_TYPE_DIR:
    case FILE_TYPE_BLENDERLIB:
      throw std::exception();
  }
}

void FileEntry::setAlias(const FileAliasInfo &alias_info)
{
  redirect_path_ = alias_info.redirection_path;
}

bool FileEntry::isAlias() const
{
  return bool(redirect_path_);
}

DirectoryEntry::DirectoryEntry(const bli_direntry &direntry,
                               const eFileAttributes attributes,
                               const AbstractFileEntry *parent)
    : AbstractFileEntry(direntry, attributes, parent)
{
}

bool DirectoryEntry::canBeEntered(const Main &, const FileList &) const
{
  /* Regular directories can always be entered. */
  /* TODO should this return false for invalid links? */
  return true;
}

void DirectoryEntry::setAlias(const FileAliasInfo &alias_info)
{
  redirect_path_ = alias_info.redirection_path;

  /* Ensure final '/'. */
  redirect_path_->resize(FILE_MAXDIR);
  BLI_path_slash_ensure(redirect_path_->data());
  redirect_path_->shrink_to_fit();
}

bool DirectoryEntry::isAlias() const
{
  return redirect_path_ == std::nullopt;
}

FileEntires &DirectoryEntry::children()
{
  return children_;
}

BlendEntry::BlendEntry(const bli_direntry &direntry,
                       const eFileAttributes attributes,
                       const AbstractFileEntry *parent)
    : DirectoryEntry(direntry, attributes, parent)
{
}

bool BlendEntry::canBeEntered(const Main &current_main, const FileList &file_list) const
{
  std::string file_path = file_list.fullFilePathToFile(*this);

  StringRef main_path = BKE_main_blendfile_path(&current_main);

  BLI_assert(file_path.empty() == false);
  BLI_assert(main_path.is_empty() == false);
  BLI_assert(file_path.back() != SEP);
  BLI_assert(main_path.back() != SEP);

  /* Maybe should be stored when reading? Would require passing, main though, which isn't great. */
  return BLI_path_cmp(file_path.data(), main_path.data()) != 0;
}

BlendIDEntry::BlendIDEntry(const bli_direntry &direntry,
                           const eFileAttributes attributes,
                           const AbstractFileEntry &parent)
    : AbstractFileEntry(direntry, attributes, &parent)
{
}

IDAssetEntry::IDAssetEntry(const bli_direntry &direntry,
                           const eFileAttributes attributes,
                           const AbstractFileEntry &parent)
    : BlendIDEntry(direntry, attributes, parent)
{
}

}  // namespace blender::ed::filelist
