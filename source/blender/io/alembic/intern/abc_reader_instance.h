/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

/** \file
 * \ingroup balembic
 */

#include "abc_reader_object.h"

namespace blender::io::alembic {

class AbcInstanceReader final : public AbcObjectReader {
 public:
  AbcInstanceReader(const Alembic::Abc::IObject &object, ImportSettings &settings);

  bool valid() const override;

  bool accepts_object_type(const Alembic::AbcCoreAbstract::ObjectHeader &alembic_header,
                           const Object *const ob,
                           const char **err_str) const override;

  void readObjectData(Main *bmain,
                      const AbcReaderManager &manager,
                      const Alembic::Abc::ISampleSelector &sample_sel) override;
};

}  // namespace blender::io::alembic
