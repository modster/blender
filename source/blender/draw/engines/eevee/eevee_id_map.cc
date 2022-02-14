/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * Structures to identify unique data blocks. The keys are unique so we are able to
 * match ids across frame updates.
 */

#include "eevee_instance.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Draw Data
 *
 * \{ */

static void draw_data_init_cb(struct DrawData *dd)
{
  /* Object has just been created or was never evaluated by the engine. */
  dd->recalc = ID_RECALC_ALL;
}

ObjectHandle &SyncModule::sync_object(Object *ob)
{
  DrawEngineType *owner = (DrawEngineType *)&DRW_engine_viewport_eevee_type;
  struct DrawData *dd = DRW_drawdata_ensure(
      (ID *)ob, owner, sizeof(eevee::ObjectHandle), draw_data_init_cb, nullptr);
  ObjectHandle &eevee_dd = *reinterpret_cast<ObjectHandle *>(dd);

  if (eevee_dd.object_key.ob == nullptr) {
    eevee_dd.object_key = ObjectKey(ob);
  }

  const int recalc_flags = ID_RECALC_COPY_ON_WRITE | ID_RECALC_TRANSFORM | ID_RECALC_SHADING |
                           ID_RECALC_GEOMETRY;
  if ((eevee_dd.recalc & recalc_flags) != 0) {
    inst_.sampling.reset();
  }

  return eevee_dd;
}

WorldHandle &SyncModule::sync_world(::World *world)
{
  DrawEngineType *owner = (DrawEngineType *)&DRW_engine_viewport_eevee_type;
  struct DrawData *dd = DRW_drawdata_ensure(
      (ID *)world, owner, sizeof(eevee::WorldHandle), draw_data_init_cb, nullptr);
  WorldHandle &eevee_dd = *reinterpret_cast<WorldHandle *>(dd);

  const int recalc_flags = ID_RECALC_ALL;
  if ((eevee_dd.recalc & recalc_flags) != 0) {
    inst_.sampling.reset();
  }
  return eevee_dd;
}

/** \} */
}  // namespace blender::eevee
