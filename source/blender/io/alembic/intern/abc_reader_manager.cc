/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup balembic
 */

#include "abc_reader_manager.h"

#include "abc_reader_instance.h"

namespace blender::io::alembic {

AbcObjectReader *AbcReaderManager::create_instance_reader(Alembic::Abc::v12::IObject iobject,
                                                          ImportSettings &settings)
{
  std::cerr << "Creating an instance reader...\n";
  AbcObjectReader *reader = new AbcInstanceReader(iobject, settings);
  m_instance_readers.push_back(reader);
  m_readers_all.push_back(reader);
  return reader;
}

Object *AbcReaderManager::get_blender_object_for_path(const std::string &path) const
{
  AbcObjectReader *reader = get_object_reader_for_path(path);
  if (!reader) {
    return nullptr;
  }
  return reader->object();
}

AbcObjectReader *AbcReaderManager::get_object_reader_for_path(const std::string &path) const
{
  MapIteratorType iter = m_readers_map.find(path);
  if (iter == m_readers_map.end()) {
    return nullptr;
  }
  return iter->second;
}

}  // namespace blender::io::alembic
