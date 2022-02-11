/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

/** \file
 * \ingroup balembic
 */

#include "abc_reader_object.h"

namespace blender::io::alembic {

class AbcReaderManager {
  using MapType = std::map<std::string, AbcObjectReader *>;
  using MapIteratorType = MapType::const_iterator;
  std::map<std::string, AbcObjectReader *> m_readers_map{};

  AbcObjectReader::ptr_vector m_readers;
  AbcObjectReader::ptr_vector m_readers_all;
  AbcObjectReader::ptr_vector m_instance_readers;

 public:
  template<typename ReaderType>
  AbcObjectReader *create(Alembic::Abc::IObject iobject, ImportSettings &settings)
  {
    static_assert(
        std::is_base_of_v<AbcObjectReader, ReaderType>,
        "Trying to create a reader from a class which does not derive from AbcObjectReader !");

    if (iobject.isInstanceRoot()) {
      return create_instance_reader(iobject, settings);
    }

    ReaderType *reader = new ReaderType(iobject, settings);
    m_readers_map[iobject.getFullName()] = reader;
    m_readers.push_back(reader);
    m_readers_all.push_back(reader);
    return reader;
  }

  AbcObjectReader *create_instance_reader(Alembic::Abc::IObject iobject, ImportSettings &settings);

  Object *get_blender_object_for_path(const std::string &path) const;

  const AbcObjectReader::ptr_vector &all_readers() const
  {
    return m_readers_all;
  }

  const AbcObjectReader::ptr_vector &instance_readers() const
  {
    return m_instance_readers;
  }

  const AbcObjectReader::ptr_vector &data_readers() const
  {
    return m_readers;
  }

 private:
  AbcObjectReader *get_object_reader_for_path(const std::string &path) const;
};

}  // namespace blender::io::alembic
