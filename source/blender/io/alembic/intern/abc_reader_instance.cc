/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "abc_reader_instance.h"

#include "DNA_object_types.h"

#include "BLI_assert.h"

#include "BKE_lib_id.h"
#include "BKE_lib_remap.h"
#include "BKE_object.h"

#include "abc_reader_manager.h"

namespace blender::io::alembic {

AbcInstanceReader::AbcInstanceReader(const Alembic::Abc::IObject &object, ImportSettings &settings)
    : AbcObjectReader(object, settings)
{
}

bool AbcInstanceReader::valid() const
{
  // TODO(kevindietrich)
  return true;
}

bool AbcInstanceReader::accepts_object_type(
    const Alembic::AbcCoreAbstract::ObjectHeader & /*alembic_header*/,
    const Object *const /*ob*/,
    const char ** /*err_str*/) const
{
  /* TODO(kevindietrich) */
  return true;
}

void AbcInstanceReader::readObjectData(Main *bmain,
                                       const AbcReaderManager &manager,
                                       const Alembic::Abc::ISampleSelector & /* sample_sel */)
{
  /* For reference on duplication, see ED_object_add_duplicate_linked.
   *
   * In this function, we only duplicate the object, as the rest (adding to view layer, tagging the
   * depsgraph, etc.) is done at the end of the import.
   */

  Object *ob = manager.get_blender_object_for_path(m_iobject.instanceSourcePath());
  BLI_assert(ob);
  /* 0 = linked */
  uint dupflag = 0;
  uint duplicate_options = LIB_ID_DUPLICATE_IS_SUBPROCESS | LIB_ID_DUPLICATE_IS_ROOT_ID;

  m_object = static_cast<Object *>(
      ID_NEW_SET(ob, BKE_object_duplicate(bmain, ob, dupflag, duplicate_options)));

  /* link own references to the newly duplicated data T26816. */
  BKE_libblock_relink_to_newid(bmain, &m_object->id, 0);
}

}  // namespace blender::io::alembic
