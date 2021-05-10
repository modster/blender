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
#include "BLI_path_util.h"

#include "file_list_entry.hh"

namespace blender::ed::filelist {

AbstractFileListEntry::AbstractFileListEntry(const bli_direntry &direntry,
                                             const eFileAttributes attributes,
                                             const AbstractFileListEntry *parent)
    : name_(direntry.relname),
      /* Could get the attributes via `BLI_file_attributes(direntry.path)` here, but it's needed
         earlier already and it's not too cheap (bunch of file system queries). */
      attributes_(attributes),
      stat_(direntry.s),
      parent_(parent)
{
}

void AbstractFileListEntry::hide()
{
  attributes_ |= FILE_ATTR_HIDDEN;
}
bool AbstractFileListEntry::isHidden() const
{
  return attributes_ & FILE_ATTR_HIDDEN;
}

StringRef AbstractFileListEntry::name() const
{
  return name_;
}

eFileAttributes AbstractFileListEntry::attributes() const
{
  return attributes_;
}

const BLI_stat_t &AbstractFileListEntry::stat() const
{
  return stat_;
}

void AbstractFileListEntry::relative_path(char *path, size_t maxsize) const
{
  BLI_assert(maxsize > 0);
  path[0] = 0;

  Vector<const char *, 10> path_vec;
  for (const AbstractFileListEntry *parent = parent_; parent != nullptr;
       parent = parent->parent_) {
    path_vec.append(parent->name_.data());
  }

  for (const char *entry : path_vec) {
    BLI_path_append(path, maxsize, entry);
  }
}

std::string AbstractFileListEntry::relative_path() const
{
  char path[PATH_MAX];
  relative_path(path, sizeof(path));
  return path;
}

std::string AbstractFileListEntry::relative_file_path() const
{
  char path[PATH_MAX];
  relative_path(path, sizeof(path));
  BLI_path_append(path, sizeof(path), name_.data());
  return path;
}

FileEntry::FileEntry(const bli_direntry &direntry,
                     const eFileAttributes attributes,
                     const AbstractFileListEntry *parent)
    : AbstractFileListEntry(direntry, attributes, parent)
{
}

void FileEntry::setAlias(const FileAliasInfo &alias_info)
{
  redirect_path_ = alias_info.redirection_path;

  if (alias_info.broken) {
    attributes_ |= FILE_ATTR_HIDDEN;
  }
}

DirectoryEntry::DirectoryEntry(const bli_direntry &direntry,
                               const eFileAttributes attributes,
                               const AbstractFileListEntry *parent)
    : AbstractFileListEntry(direntry, attributes, parent)
{
}

void DirectoryEntry::setAlias(const FileAliasInfo &alias_info)
{
  redirect_path_ = alias_info.redirection_path;

  /* Ensure final '/'. */
  redirect_path_->resize(FILE_MAXDIR);
  BLI_path_slash_ensure(redirect_path_->data());
  redirect_path_->shrink_to_fit();

  if (alias_info.broken) {
    attributes_ |= FILE_ATTR_HIDDEN;
  }
}

bool DirectoryEntry::isAlias() const
{
  return redirect_path_ == std::nullopt;
}

FileTree &DirectoryEntry::children()
{
  return children_;
}

BlendEntry::BlendEntry(const bli_direntry &direntry,
                       const eFileAttributes attributes,
                       const AbstractFileListEntry *parent)
    : DirectoryEntry(direntry, attributes, parent)
{
}

BlendIDEntry::BlendIDEntry(const bli_direntry &direntry,
                           const eFileAttributes attributes,
                           const AbstractFileListEntry &parent)
    : AbstractFileListEntry(direntry, attributes, &parent)
{
}

IDAssetEntry::IDAssetEntry(const bli_direntry &direntry,
                           const eFileAttributes attributes,
                           const AbstractFileListEntry &parent)
    : BlendIDEntry(direntry, attributes, parent)
{
}

}  // namespace blender::ed::filelist
