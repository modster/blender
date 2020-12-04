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
 *
 * The Original Code is Copyright (C) 2019 Blender Foundation.
 * All rights reserved.
 */

#include "import/usd_data_cache.h"
#include "import/usd_importer_context.h"
#include "import/usd_prim_iterator.h"
#include "import/usd_reader_mesh_base.h"
#include "import/usd_reader_xformable.h"
#include "usd.h"
#include "usd_hierarchy_iterator.h"

#include <pxr/base/plug/registry.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>

#include "MEM_guardedalloc.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_build.h"
#include "DEG_depsgraph_query.h"

#include "DNA_scene_types.h"

#include "BKE_appdir.h"
#include "BKE_blender_version.h"
#include "BKE_context.h"
#include "BKE_global.h"
#include "BKE_scene.h"

// XXX check the following
#include "BKE_main.h"
#include "BKE_material.h"
#include "BKE_mesh.h"
#include "BKE_modifier.h"
#include "BKE_object.h"
// XXX check the following
#include "BKE_cachefile.h"
#include "BKE_context.h"
#include "BKE_curve.h"
#include "BKE_global.h"
#include "BKE_layer.h"
#include "BKE_lib_id.h"
#include "BKE_object.h"
#include "BKE_scene.h"
#include "BKE_screen.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BLI_fileops.h"
#include "BLI_path_util.h"
#include "BLI_string.h"

#include "ED_undo.h"

#include "WM_api.h"
#include "WM_types.h"

#include <iostream>

namespace blender::io::usd {

struct ExportJobData {
  Main *bmain;
  Depsgraph *depsgraph;
  wmWindowManager *wm;

  char filename[FILE_MAX];
  USDExportParams params;

  bool export_ok;
};

static void ensure_usd_plugin_path_registered(void)
{
  static bool plugin_path_registered = false;
  if (plugin_path_registered) {
    return;
  }
  plugin_path_registered = true;

  /* Tell USD which directory to search for its JSON files. If 'datafiles/usd'
   * does not exist, the USD library will not be able to read or write any files. */
  const std::string blender_usd_datafiles = BKE_appdir_folder_id(BLENDER_DATAFILES, "usd");
  /* The trailing slash indicates to the USD library that the path is a directory. */
  pxr::PlugRegistry::GetInstance().RegisterPlugins(blender_usd_datafiles + "/");
}

static void export_startjob(void *customdata,
                            /* Cannot be const, this function implements wm_jobs_start_callback.
                             * NOLINTNEXTLINE: readability-non-const-parameter. */
                            short *stop,
                            short *do_update,
                            float *progress)
{
  ExportJobData *data = static_cast<ExportJobData *>(customdata);
  data->export_ok = false;

  G.is_rendering = true;
  WM_set_locked_interface(data->wm, true);
  G.is_break = false;

  // Construct the depsgraph for exporting.
  Scene *scene = DEG_get_input_scene(data->depsgraph);
  if (data->params.visible_objects_only) {
    DEG_graph_build_from_view_layer(data->depsgraph);
  }
  else {
    DEG_graph_build_for_all_objects(data->depsgraph);
  }
  BKE_scene_graph_update_tagged(data->depsgraph, data->bmain);

  *progress = 0.0f;
  *do_update = true;

  // For restoring the current frame after exporting animation is done.
  const int orig_frame = CFRA;

  pxr::UsdStageRefPtr usd_stage = pxr::UsdStage::CreateNew(data->filename);
  if (!usd_stage) {
    /* This happens when the USD JSON files cannot be found. When that happens,
     * the USD library doesn't know it has the functionality to write USDA and
     * USDC files, and creating a new UsdStage fails. */
    WM_reportf(
        RPT_ERROR, "USD Export: unable to find suitable USD plugin to write %s", data->filename);
    return;
  }

  usd_stage->SetMetadata(pxr::UsdGeomTokens->upAxis, pxr::VtValue(pxr::UsdGeomTokens->z));
  usd_stage->SetMetadata(pxr::UsdGeomTokens->metersPerUnit,
                         pxr::VtValue(scene->unit.scale_length));
  usd_stage->GetRootLayer()->SetDocumentation(std::string("Blender v") +
                                              BKE_blender_version_string());

  // Set up the stage for animated data.
  if (data->params.export_animation) {
    usd_stage->SetTimeCodesPerSecond(FPS);
    usd_stage->SetStartTimeCode(scene->r.sfra);
    usd_stage->SetEndTimeCode(scene->r.efra);
  }

  USDHierarchyIterator iter(data->depsgraph, usd_stage, data->params);

  if (data->params.export_animation) {
    // Writing the animated frames is not 100% of the work, but it's our best guess.
    float progress_per_frame = 1.0f / std::max(1, (scene->r.efra - scene->r.sfra + 1));

    for (float frame = scene->r.sfra; frame <= scene->r.efra; frame++) {
      if (G.is_break || (stop != nullptr && *stop)) {
        break;
      }

      // Update the scene for the next frame to render.
      scene->r.cfra = static_cast<int>(frame);
      scene->r.subframe = frame - scene->r.cfra;
      BKE_scene_graph_update_for_newframe(data->depsgraph);

      iter.set_export_frame(frame);
      iter.iterate_and_write();

      *progress += progress_per_frame;
      *do_update = true;
    }
  }
  else {
    // If we're not animating, a single iteration over all objects is enough.
    iter.iterate_and_write();
  }

  iter.release_writers();
  usd_stage->GetRootLayer()->Save();

  // Finish up by going back to the keyframe that was current before we started.
  if (CFRA != orig_frame) {
    CFRA = orig_frame;
    BKE_scene_graph_update_for_newframe(data->depsgraph);
  }

  data->export_ok = true;
  *progress = 1.0f;
  *do_update = true;
}

static void export_endjob(void *customdata)
{
  ExportJobData *data = static_cast<ExportJobData *>(customdata);

  DEG_graph_free(data->depsgraph);

  if (!data->export_ok && BLI_exists(data->filename)) {
    BLI_delete(data->filename, false, false);
  }

  G.is_rendering = false;
  WM_set_locked_interface(data->wm, false);
}

struct ImportJobData {
  bContext *C;
  Main *bmain;
  Scene *scene;
  ViewLayer *view_layer;
  wmWindowManager *wm;

  char filename[1024];

  USDImportParams params;

  short *stop;
  short *do_update;
  float *progress;

  bool was_cancelled;
  bool import_ok;
  bool is_background_job;

  pxr::UsdStageRefPtr stage;
  std::vector<USDXformableReader *> readers;
  USDDataCache data_cache;
};

static void import_startjob(void *user_data, short *stop, short *do_update, float *progress)
{
  ImportJobData *data = static_cast<ImportJobData *>(user_data);

  data->import_ok = false;
  data->stop = stop;
  data->do_update = do_update;
  data->progress = progress;

  Scene *scene = data->scene;

  WM_set_locked_interface(data->wm, true);

  *data->do_update = true;
  *data->progress = 0.05f;

  data->stage = pxr::UsdStage::Open(data->filename);
  if (!data->stage) {
    WM_reportf(RPT_ERROR, "USD Export: couldn't open USD stage for file %s", data->filename);
    return;
  }

  pxr::TfToken up_axis = pxr::UsdGeomGetStageUpAxis(data->stage);
  USDImporterContext import_ctx{up_axis, data->params};

  USDPrimIterator usd_prim_iter(data->stage, import_ctx, data->bmain);

  // Optionally print the stage contents for debugging.
  if (data->params.debug) {
    usd_prim_iter.debug_traverse_stage();
  }

  if (G.is_break) {
    data->was_cancelled = true;
    return;
  }

  *data->do_update = true;
  *data->progress = 0.1f;

  /* Optionally, cache the prototype data for instancing. */
  if (data->params.use_instancing) {
    usd_prim_iter.cache_prototype_data(data->data_cache);
  }

  /* Get the xformable prim readers. */
  usd_prim_iter.create_object_readers(data->readers);

  // Create objects

  const float size = static_cast<float>(data->readers.size());
  size_t i = 0;

  double time = CFRA;

  std::vector<USDXformableReader *>::iterator iter;
  for (iter = data->readers.begin(); iter != data->readers.end(); ++iter) {
    USDXformableReader *reader = *iter;

    if (reader->valid()) {
      reader->create_object(data->bmain, time, &data->data_cache);
    }
    else {
      std::cerr << "Object " << reader->prim_path() << " in USD file " << data->filename
                << " is invalid.\n";
    }

    *data->progress = 0.1f + 0.3f * (++i / size);
    *data->do_update = true;

    if (G.is_break) {
      data->was_cancelled = true;
      return;
    }
  }

  /* Setup parenthood. */
  for (iter = data->readers.begin(); iter != data->readers.end(); ++iter) {
    const USDXformableReader *reader = *iter;

    Object *ob = reader->object();

    if (!ob) {
      continue;
    }

    const USDXformableReader *parent_reader = reader->parent();

    ob->parent = parent_reader ? parent_reader->object() : nullptr;
  }

  /* Setup transformations. */
  i = 0;
  for (iter = data->readers.begin(); iter != data->readers.end(); ++iter) {
    USDXformableReader *reader = *iter;
    reader->set_object_transform(time);

    *data->progress = 0.7f + 0.3f * (++i / size);
    *data->do_update = true;

    if (G.is_break) {
      data->was_cancelled = true;
      return;
    }
  }
}

static void import_endjob(void *user_data)
{
  ImportJobData *data = static_cast<ImportJobData *>(user_data);

  std::vector<USDXformableReader *>::iterator iter;

  /* Delete objects on cancellation. */
  if (data->was_cancelled) {
    for (iter = data->readers.begin(); iter != data->readers.end(); ++iter) {
      Object *ob = (*iter)->object();

      /* It's possible that cancellation occurred between the creation of
       * the reader and the creation of the Blender object. */
      if (ob != NULL) {
        BKE_id_free_us(data->bmain, ob);
      }
    }
  }
  else {
    /* Add object to scene. */
    Base *base;
    LayerCollection *lc;
    ViewLayer *view_layer = data->view_layer;

    BKE_view_layer_base_deselect_all(view_layer);

    lc = BKE_layer_collection_get_active(view_layer);

    for (iter = data->readers.begin(); iter != data->readers.end(); ++iter) {
      Object *ob = (*iter)->object();

      if (!ob) {
        continue;
      }

      BKE_collection_object_add(data->bmain, lc->collection, ob);

      base = BKE_view_layer_base_find(view_layer, ob);
      /* TODO: is setting active needed? */
      BKE_view_layer_base_select_and_set_active(view_layer, base);

      DEG_id_tag_update(&lc->collection->id, ID_RECALC_COPY_ON_WRITE);
      DEG_id_tag_update_ex(data->bmain,
                           &ob->id,
                           ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY | ID_RECALC_ANIMATION |
                               ID_RECALC_BASE_FLAGS);
    }

    DEG_id_tag_update(&data->scene->id, ID_RECALC_BASE_FLAGS);
    DEG_relations_tag_update(data->bmain);

    if (data->is_background_job) {
      /* Blender already returned from the import operator, so we need to store our own extra undo
       * step. */
      ED_undo_push(data->C, "USD Import Finished");
    }
  }

  /* Delete the reders. */

  for (iter = data->readers.begin(); iter != data->readers.end(); ++iter) {
    delete *iter;
  }

  data->readers.clear();

  /* TODO(makowalski): Explicitly clear the data cache as well? */

  WM_set_locked_interface(data->wm, false);

  data->import_ok = !data->was_cancelled;

  WM_main_add_notifier(NC_SCENE | ND_FRAME, data->scene);
}

static void import_freejob(void *user_data)
{
  ImportJobData *data = static_cast<ImportJobData *>(user_data);
  delete data;
}

}  // namespace blender::io::usd

bool USD_export(bContext *C,
                const char *filepath,
                const USDExportParams *params,
                bool as_background_job)
{
  ViewLayer *view_layer = CTX_data_view_layer(C);
  Scene *scene = CTX_data_scene(C);

  blender::io::usd::ensure_usd_plugin_path_registered();

  blender::io::usd::ExportJobData *job = static_cast<blender::io::usd::ExportJobData *>(
      MEM_mallocN(sizeof(blender::io::usd::ExportJobData), "ExportJobData"));

  job->bmain = CTX_data_main(C);
  job->wm = CTX_wm_manager(C);
  job->export_ok = false;
  BLI_strncpy(job->filename, filepath, sizeof(job->filename));

  job->depsgraph = DEG_graph_new(job->bmain, scene, view_layer, params->evaluation_mode);
  job->params = *params;

  bool export_ok = false;
  if (as_background_job) {
    wmJob *wm_job = WM_jobs_get(
        job->wm, CTX_wm_window(C), scene, "USD Export", WM_JOB_PROGRESS, WM_JOB_TYPE_ALEMBIC);

    /* setup job */
    WM_jobs_customdata_set(wm_job, job, MEM_freeN);
    WM_jobs_timer(wm_job, 0.1, NC_SCENE | ND_FRAME, NC_SCENE | ND_FRAME);
    WM_jobs_callbacks(wm_job,
                      blender::io::usd::export_startjob,
                      nullptr,
                      nullptr,
                      blender::io::usd::export_endjob);

    WM_jobs_start(CTX_wm_manager(C), wm_job);
  }
  else {
    /* Fake a job context, so that we don't need NULL pointer checks while exporting. */
    short stop = 0, do_update = 0;
    float progress = 0.f;

    blender::io::usd::export_startjob(job, &stop, &do_update, &progress);
    blender::io::usd::export_endjob(job);
    export_ok = job->export_ok;

    MEM_freeN(job);
  }

  return export_ok;
}

bool USD_import(bContext *C,
                const char *filepath,
                const struct USDImportParams *params,
                bool as_background_job)
{
  blender::io::usd::ensure_usd_plugin_path_registered();

  /* Using new here since MEM_* functions do not call constructor to properly initialize data. */
  blender::io::usd::ImportJobData *job = new blender::io::usd::ImportJobData();
  job->C = C;
  job->bmain = CTX_data_main(C);
  job->scene = CTX_data_scene(C);
  job->view_layer = CTX_data_view_layer(C);
  job->wm = CTX_wm_manager(C);
  job->import_ok = false;
  BLI_strncpy(job->filename, filepath, 1024);
  job->params = *params;

  job->was_cancelled = false;
  job->is_background_job = as_background_job;

  G.is_break = false;

  bool import_ok = false;

  if (as_background_job) {
    wmJob *wm_job = WM_jobs_get(job->wm,
                                CTX_wm_window(C),
                                job->scene,
                                "USD Import",
                                WM_JOB_PROGRESS,
                                WM_JOB_TYPE_ALEMBIC);  // XXX -- Here and above, why TYPE_ALEMBIC?

    /* setup job */
    WM_jobs_customdata_set(wm_job, job, blender::io::usd::import_freejob);
    WM_jobs_timer(wm_job, 0.1, NC_SCENE | ND_FRAME, NC_SCENE | ND_FRAME);
    WM_jobs_callbacks(wm_job,
                      blender::io::usd::import_startjob,
                      nullptr,
                      nullptr,
                      blender::io::usd::import_endjob);

    WM_jobs_start(CTX_wm_manager(C), wm_job);
  }
  else {
    /* Fake a job context, so that we don't need NULL pointer checks while importing. */
    short stop = 0, do_update = 0;
    float progress = 0.f;

    blender::io::usd::import_startjob(job, &stop, &do_update, &progress);
    blender::io::usd::import_endjob(job);
    import_ok = job->import_ok;

    blender::io::usd::import_freejob(job);
  }

  return import_ok;
}

int USD_get_version(void)
{
  /* USD 20.05 defines:
   *
   * #define PXR_MAJOR_VERSION 0
   * #define PXR_MINOR_VERSION 20
   * #define PXR_PATCH_VERSION 05
   * #define PXR_VERSION 2005
   *
   * So the major version is implicit/invisible in the public version number.
   */
  return PXR_VERSION;
}
