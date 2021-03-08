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

#include "usd.h"
#include "usd_hierarchy_iterator.h"
#include "usd_reader_instance.h"
#include "usd_reader_mesh.h"
#include "usd_reader_prim.h"
#include "usd_reader_stage.h"

#include <pxr/base/plug/registry.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>

#include "MEM_guardedalloc.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_build.h"
#include "DEG_depsgraph_query.h"

#include "DNA_cachefile_types.h"
#include "DNA_collection_types.h"
#include "DNA_node_types.h"
#include "DNA_scene_types.h"
#include "DNA_world_types.h"

#include "BKE_appdir.h"
#include "BKE_blender_version.h"
#include "BKE_cachefile.h"
#include "BKE_cdderivedmesh.h"
#include "BKE_context.h"
#include "BKE_curve.h"
#include "BKE_global.h"
#include "BKE_layer.h"
#include "BKE_lib_id.h"
#include "BKE_library.h"
#include "BKE_main.h"
#include "BKE_node.h"
#include "BKE_object.h"
#include "BKE_scene.h"
#include "BKE_world.h"

#include "BLI_fileops.h"
#include "BLI_listbase.h"
#include "BLI_math_matrix.h"
#include "BLI_math_rotation.h"
#include "BLI_path_util.h"
#include "BLI_string.h"

#include "WM_api.h"
#include "WM_types.h"

#include "usd_reader_geom.h"
#include "usd_reader_prim.h"
#include "usd_util.h"

#include <iostream>

struct USDStageHandle {
  CacheArchiveHandle base;
};

namespace blender::io::usd {

USDStageReader *archive_from_handle(USDStageHandle *handle)
{
  return reinterpret_cast<USDStageReader *>(handle);
}

USDStageReader *archive_from_handle(CacheArchiveHandle *handle)
{
  return reinterpret_cast<USDStageReader *>(handle);
}

USDStageHandle *handle_from_archive(USDStageReader *archive)
{
  return reinterpret_cast<USDStageHandle *>(archive);
}

static bool gather_objects_paths(const pxr::UsdPrim &object, ListBase *object_paths)
{
  if (!object.IsValid()) {
    return false;
  }

  for (const pxr::UsdPrim &childPrim : object.GetChildren()) {
    gather_objects_paths(childPrim, object_paths);
  }

  void *usd_path_void = MEM_callocN(sizeof(CacheObjectPath), "CacheObjectPath");
  CacheObjectPath *usd_path = static_cast<CacheObjectPath *>(usd_path_void);

  BLI_strncpy(usd_path->path, object.GetPrimPath().GetString().c_str(), sizeof(usd_path->path));
  BLI_addtail(object_paths, usd_path);

  return true;
}

/* Create a collection with the given parent and name. */
static Collection *create_collection(Main *bmain, Collection *parent, const char *name)
{
  if (!bmain) {
    return nullptr;
  }

  Collection *coll = BKE_collection_add(bmain, parent, name);

  if (coll) {
    id_fake_user_set(&coll->id);
    DEG_id_tag_update(&coll->id, ID_RECALC_COPY_ON_WRITE);
  }

  return coll;
}

/* Set the instance collection on the given instance reader.
 *  The collection is assigned from the given map based on
 *  the prototype (maser) prim path. */
static void set_instance_collection(
    USDInstanceReader *instance_reader,
    const std::map<pxr::SdfPath, Collection *> &proto_collection_map)
{
  if (!instance_reader) {
    return;
  }

  pxr::SdfPath proto_path = instance_reader->proto_path();

  std::map<pxr::SdfPath, Collection *>::const_iterator it = proto_collection_map.find(proto_path);

  if (it != proto_collection_map.end()) {
    instance_reader->set_instance_collection(it->second);
  }
  else {
    std::cerr << "WARNING: Couldn't find prototype collection for " << instance_reader->prim_path()
              << std::endl;
  }
}

/* Create instance collections for the USD instance readers. */
static void create_proto_collections(Main *bmain,
                                     ViewLayer *view_layer,
                                     Collection *parent_collection,
                                     const USDStageReader::ProtoReaderMap &proto_readers,
                                     const std::vector<USDPrimReader *> &readers)
{
  Collection *all_protos_collection = create_collection(bmain, parent_collection, "prototypes");

  std::map<pxr::SdfPath, Collection *> proto_collection_map;

  for (const auto &pair : proto_readers) {

    std::string proto_collection_name = pair.first.GetString();

    // TODO(makowalski): Is it acceptable to have slashes in the collection names? Or should we
    // replace them with another character, like an underscore, as in the following?
    // std::replace(proto_collection_name.begin(), proto_collection_name.end(), '/', '_');

    Collection *proto_collection = create_collection(
        bmain, all_protos_collection, proto_collection_name.c_str());

    LayerCollection *proto_lc = BKE_layer_collection_first_from_scene_collection(view_layer,
                                                                                 proto_collection);
    if (proto_lc) {
      proto_lc->flag |= LAYER_COLLECTION_HIDE;
    }

    proto_collection_map.insert(std::make_pair(pair.first, proto_collection));
  }

  // Set the instance collections on the readers, including the prototype
  // readers, as instancing may be recursive.

  for (const auto &pair : proto_readers) {
    for (USDPrimReader *reader : pair.second) {
      if (USDInstanceReader *instance_reader = dynamic_cast<USDInstanceReader *>(reader)) {
        set_instance_collection(instance_reader, proto_collection_map);
      }
    }
  }

  for (USDPrimReader *reader : readers) {
    if (USDInstanceReader *instance_reader = dynamic_cast<USDInstanceReader *>(reader)) {
      set_instance_collection(instance_reader, proto_collection_map);
    }
  }

  // Add the prototype objects to the collections.
  for (const auto &pair : proto_readers) {

    std::map<pxr::SdfPath, Collection *>::const_iterator it = proto_collection_map.find(
        pair.first);

    if (it == proto_collection_map.end()) {
      std::cerr << "WARNING: Couldn't find collection when adding objects for prototype "
                << pair.first << std::endl;
      continue;
    }

    for (USDPrimReader *reader : pair.second) {
      Object *ob = reader->object();

      if (!ob) {
        continue;
      }

      Collection *coll = it->second;

      BKE_collection_object_add(bmain, coll, ob);

      DEG_id_tag_update(&coll->id, ID_RECALC_COPY_ON_WRITE);
      DEG_id_tag_update_ex(bmain,
                           &ob->id,
                           ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY | ID_RECALC_ANIMATION |
                               ID_RECALC_BASE_FLAGS);
    }
  }
}

// Update the given import settings with the global rotation matrix to orient
// imported objects with Z-up, if necessary
static void set_global_rotation(pxr::UsdStageRefPtr stage, ImportSettings &r_settings)
{
  if (!stage || pxr::UsdGeomGetStageUpAxis(stage) == pxr::UsdGeomTokens->z) {
    // Nothing to do.
    return;
  }

  r_settings.do_convert_mat = true;

  // Rotate 90 degrees about the X-axis.
  float rmat[3][3];
  float axis[3] = {1.0f, 0.0f, 0.0f};
  axis_angle_normalized_to_mat3(rmat, axis, M_PI / 2.0f);

  unit_m4(r_settings.conversion_mat);
  copy_m4_m3(r_settings.conversion_mat, rmat);
}

/* ********************** Export file ********************** */

struct ExportJobData {
  Main *bmain;
  Depsgraph *depsgraph;
  wmWindowManager *wm;

  char filename[FILE_MAX];
  USDExportParams params;

  bool export_ok;
};

static void ensure_usd_plugin_path_registered()
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

  /* Construct the depsgraph for exporting. */
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

  /* For restoring the current frame after exporting animation is done. */
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

  /* Set up the stage for animated data. */
  if (data->params.export_animation) {
    usd_stage->SetTimeCodesPerSecond(FPS);
    usd_stage->SetStartTimeCode(scene->r.sfra);
    usd_stage->SetEndTimeCode(scene->r.efra);
  }

  USDHierarchyIterator iter(data->depsgraph, usd_stage, data->params);

  if (data->params.export_animation) {
    /* Writing the animated frames is not 100% of the work, but it's our best guess. */
    float progress_per_frame = 1.0f / std::max(1, (scene->r.efra - scene->r.sfra + 1));

    for (float frame = scene->r.sfra; frame <= scene->r.efra; frame++) {
      if (G.is_break || (stop != nullptr && *stop)) {
        break;
      }

      /* Update the scene for the next frame to render. */
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
    /* If we're not animating, a single iteration over all objects is enough. */
    iter.iterate_and_write();
  }

  iter.release_writers();
  usd_stage->GetRootLayer()->Save();

  /* Finish up by going back to the keyframe that was current before we started. */
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
    // TODO: Should create WM_JOB_TYPE_USD...

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
    float progress = 0.0f;

    blender::io::usd::export_startjob(job, &stop, &do_update, &progress);
    blender::io::usd::export_endjob(job);
    export_ok = job->export_ok;

    MEM_freeN(job);
  }

  return export_ok;
}

int USD_get_version(void)
{
  /* USD 19.11 defines:
   *
   * #define PXR_MAJOR_VERSION 0
   * #define PXR_MINOR_VERSION 19
   * #define PXR_PATCH_VERSION 11
   * #define PXR_VERSION 1911
   *
   * So the major version is implicit/invisible in the public version number.
   */
  return PXR_VERSION;
}

/* ********************** Import file ********************** */

namespace blender::io::usd {
enum {
  USD_NO_ERROR = 0,
  USD_ARCHIVE_FAIL,
};

struct ImportJobData {
  Main *bmain;
  Scene *scene;
  ViewLayer *view_layer;
  wmWindowManager *wm;

  char filename[1024];
  USDImportParams params;
  ImportSettings settings;

  USDStageReader *archive;
  std::vector<USDPrimReader *> readers;

  short *stop;
  short *do_update;
  float *progress;

  char error_code;
  bool was_canceled;
  bool import_ok;
};

static void import_startjob(void *customdata, short *stop, short *do_update, float *progress)
{
  WM_reportf(RPT_WARNING, "Import start job: %s", 0);
  ImportJobData *data = static_cast<ImportJobData *>(customdata);

  data->stop = stop;
  data->do_update = do_update;
  data->progress = progress;
  data->was_canceled = false;

  // G.is_rendering = true;
  WM_set_locked_interface(data->wm, true);
  G.is_break = false;

  if (data->params.create_collection) {
    char display_name[1024];
    BLI_path_to_display_name(
        display_name, strlen(data->filename), BLI_path_basename(data->filename));
    Collection *import_collection = BKE_collection_add(
        data->bmain, data->scene->master_collection, display_name);
    id_fake_user_set(&import_collection->id);

    DEG_id_tag_update(&import_collection->id, ID_RECALC_COPY_ON_WRITE);
    DEG_relations_tag_update(data->bmain);

    WM_main_add_notifier(NC_SCENE | ND_LAYER, NULL);

    data->view_layer->active_collection = BKE_layer_collection_first_from_scene_collection(
        data->view_layer, import_collection);
  }

  BLI_path_abs(data->filename, BKE_main_blendfile_path_from_global());
  USDStageReader *archive = new USDStageReader(data->bmain, data->filename);

  archive->params(data->params);
  archive->settings(data->settings);

  CacheFile *cache_file = static_cast<CacheFile *>(
      BKE_cachefile_add(data->bmain, BLI_path_basename(data->filename)));

  /* Decrement the ID ref-count because it is going to be incremented for each
   * modifier and constraint that it will be attached to, so since currently
   * it is not used by anyone, its use count will off by one. */
  id_us_min(&cache_file->id);

  cache_file->is_sequence = data->params.is_sequence;
  cache_file->scale = data->params.scale;
  STRNCPY(cache_file->filepath, data->filename);

  data->archive = archive;
  data->settings.cache_file = cache_file;
  data->settings.vel_scale = data->params.vel_scale;

  *data->do_update = true;
  *data->progress = 0.05f;

  if (G.is_break) {
    data->was_canceled = true;
    return;
  }

  *data->do_update = true;
  *data->progress = 0.1f;

  Scene *scene = data->scene;

  if (!archive->stage()) {
    /* This happens when the USD JSON files cannot be found. When that happens,
     * the USD library doesn't know it has the functionality to write USDA and
     * USDC files, and creating a new UsdStage fails. */
    WM_reportf(
        RPT_ERROR, "USD Import: unable to find suitable USD plugin to read %s", data->filename);
    data->import_ok = false;
    return;
  }

  if (data->params.convert_to_z_up) {
    set_global_rotation(archive->stage(), data->settings);
  }

  // Set up the stage for animated data.
  if (data->params.set_frame_range) {
    // archive->stage()->SetTimeCodesPerSecond(FPS);
    scene->r.sfra = archive->stage()->GetStartTimeCode();
    scene->r.efra = archive->stage()->GetEndTimeCode();
    // archive->usd_stage->SetStartTimeCode(scene->r.sfra);
    // usd_stage->SetEndTimeCode(scene->r.efra);
  }

  *data->progress = 0.15f;

  data->readers = archive->collect_readers(data->bmain, data->params, data->settings);

  *data->progress = 0.2f;

  const float size = static_cast<float>(data->readers.size());
  size_t i = 0;

  /* Setup parenthood */

  /* Handle instance prototypes.
  /* TODO(makowalski): Move this logic inside USDReaderStage? */
  for (const auto &pair : archive->proto_readers()) {

    for (USDPrimReader *reader : pair.second) {

      if (!reader) {
        continue;
      }

      /* TODO(makowalski): Here and below, should we call
       *  readObjectData() with the actual time? */
      reader->readObjectData(data->bmain, 0.0);

      Object *ob = reader->object();

      if (!ob) {
        continue;
      }

      const USDPrimReader *parent_reader = reader->parent();

      ob->parent = parent_reader ? parent_reader->object() : nullptr;

      // TODO(makowalski): Handle progress update.
    }
  }

  std::vector<USDPrimReader *>::iterator iter;

  for (iter = data->readers.begin(); iter != data->readers.end(); ++iter) {
    Object *ob = (*iter)->object();

    USDPrimReader *reader = (*iter);
    reader->readObjectData(data->bmain, 0.0);

    USDPrimReader *parent = (*iter)->parent();

    if (parent == NULL /*|| !reader->inherits_xform()*/) {
      ob->parent = NULL;
    }
    else {
      ob->parent = parent->object();
    }

    *data->progress = 0.2f + 0.8f * (++i / size);
    *data->do_update = true;

    if (G.is_break) {
      data->was_canceled = true;
      return;
    }
  }

  data->import_ok = !data->was_canceled;

  *progress = 1.0f;
  *do_update = true;
}

static void import_endjob(void *customdata)
{
  WM_reportf(RPT_WARNING, "Import end job: %s", 0);

  ImportJobData *data = static_cast<ImportJobData *>(customdata);

  std::vector<USDPrimReader *>::iterator iter;

  /* Delete objects on cancellation. */
  if (data->was_canceled) {
    for (iter = data->readers.begin(); iter != data->readers.end(); ++iter) {
      Object *ob = (*iter)->object();

      /* It's possible that cancellation occurred between the creation of
       * the reader and the creation of the Blender object. */
      if (ob == NULL) {
        continue;
      }

      BKE_id_free_us(data->bmain, ob);
    }

    if (data->archive) {
      for (const auto &pair : data->archive->proto_readers()) {
        for (USDPrimReader *reader : pair.second) {
          Object *ob = reader->object();
          /* It's possible that cancellation occurred between the creation of
           * the reader and the creation of the Blender object. */
          if (ob != NULL) {
            BKE_id_free_us(data->bmain, ob);
          }
        }
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

    if (data->archive && !data->archive->proto_readers().empty()) {
      create_proto_collections(
          data->bmain, view_layer, lc->collection, data->archive->proto_readers(), data->readers);
    }

    for (iter = data->readers.begin(); iter != data->readers.end(); ++iter) {
      Object *ob = (*iter)->object();

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
  }

  for (iter = data->readers.begin(); iter != data->readers.end(); ++iter) {
    USDPrimReader *reader = *iter;
    reader->decref();

    if (reader->refcount() == 0) {
      delete reader;
    }
  }

  data->archive->clear_proto_readers(true);

  WM_set_locked_interface(data->wm, false);

  switch (data->error_code) {
    default:
    case USD_NO_ERROR:
      data->import_ok = !data->was_canceled;
      break;
    case USD_ARCHIVE_FAIL:
      WM_report(RPT_ERROR, "Could not open USD archive for reading! See console for detail.");
      break;
  }

  WM_main_add_notifier(NC_SCENE | ND_FRAME, data->scene);
}

static void import_freejob(void *user_data)
{
  ImportJobData *data = static_cast<ImportJobData *>(user_data);

  // delete data->archive;
  delete data;
}

}  // namespace blender::io::usd

using namespace blender::io::usd;

bool USD_import(struct bContext *C,
                const char *filepath,
                const USDImportParams *params,
                bool as_background_job)
{
  blender::io::usd::ensure_usd_plugin_path_registered();

  /* Using new here since MEM_* funcs do not call ctor to properly initialize
   * data. */
  ImportJobData *job = new ImportJobData();
  // USD::ImportJobData *job = static_cast<USD::ImportJobData
  // *>(MEM_mallocN(sizeof(USD::ImportJobData), "ImportJobData"));
  job->bmain = CTX_data_main(C);
  job->scene = CTX_data_scene(C);
  job->view_layer = CTX_data_view_layer(C);
  job->wm = CTX_wm_manager(C);
  job->import_ok = false;
  BLI_strncpy(job->filename, filepath, 1024);

  job->settings.scale = params->scale;
  job->settings.sequence_offset = params->offset;
  job->settings.is_sequence = params->is_sequence;
  job->settings.sequence_len = params->sequence_len;
  job->settings.validate_meshes = params->validate_meshes;
  job->settings.sequence_len = params->sequence_len;
  job->error_code = USD_NO_ERROR;
  job->was_canceled = false;
  job->archive = NULL;

  job->params = *params;

  G.is_break = false;

  bool import_ok = false;
  if (as_background_job) {
    wmJob *wm_job = WM_jobs_get(CTX_wm_manager(C),
                                CTX_wm_window(C),
                                job->scene,
                                "USD Import",
                                WM_JOB_PROGRESS,
                                WM_JOB_TYPE_ALEMBIC);

    /* setup job */
    WM_jobs_customdata_set(wm_job, job, import_freejob);
    WM_jobs_timer(wm_job, 0.1, NC_SCENE, NC_SCENE);
    WM_jobs_callbacks(wm_job, import_startjob, NULL, NULL, import_endjob);

    WM_jobs_start(CTX_wm_manager(C), wm_job);
  }
  else {
    /* Fake a job context, so that we don't need NULL pointer checks while importing. */
    short stop = 0, do_update = 0;
    float progress = 0.f;

    import_startjob(job, &stop, &do_update, &progress);
    import_endjob(job);
    import_ok = job->import_ok;

    import_freejob(job);
  }

  return import_ok;
}

static USDPrimReader *get_usd_reader(CacheReader *reader, Object *ob, const char **err_str)
{
  USDPrimReader *usd_reader = reinterpret_cast<USDPrimReader *>(reader);
  pxr::UsdPrim iobject = usd_reader->prim();

  if (!iobject.IsValid()) {
    *err_str = "Invalid object: verify object path";
    return NULL;
  }

  /*const ObjectHeader &header = iobject.getHeader();
  if (!usd_reader->accepts_object_type(header, ob, err_str)) {*/
  /* err_str is set by acceptsObjectType() */
  /*return NULL;
}*/

  return usd_reader;
}

Mesh *USD_read_mesh(CacheReader *reader,
                    Object *ob,
                    Mesh *existing_mesh,
                    const float time,
                    const char **err_str,
                    int read_flag,
                    float vel_fac)
{
  USDGeomReader *usd_reader = reinterpret_cast<USDGeomReader *>(
      get_usd_reader(reader, ob, err_str));
  if (usd_reader == NULL) {
    return NULL;
  }

  return usd_reader->read_mesh(existing_mesh, time, read_flag, vel_fac, err_str);
}

bool USD_mesh_topology_changed(
    CacheReader *reader, Object *ob, Mesh *existing_mesh, const float time, const char **err_str)
{
  USDMeshReader *usd_reader = (USDMeshReader *)get_usd_reader(reader, ob, err_str);

  if (usd_reader == NULL) {
    return false;
  }

  return usd_reader->topology_changed(existing_mesh, time);
}

void USDCacheReader_incref(CacheReader *reader)
{
  USDPrimReader *usd_reader = reinterpret_cast<USDPrimReader *>(reader);
  usd_reader->incref();
}

CacheReader *CacheReader_open_usd_object(USDStageHandle *handle,
                                         CacheReader *reader,
                                         Object *object,
                                         const char *object_path)
{
  if (object_path[0] == '\0') {
    return reader;
  }

  USDStageReader *archive = archive_from_handle(handle);

  if (!archive || !archive->valid()) {
    return reader;
  }

  pxr::UsdPrim prim = archive->stage()->GetPrimAtPath(pxr::SdfPath(object_path));

  if (reader) {
    USDCacheReader_free(reader);
  }

  // TODO: The handle does not have the proper import params or settings
  USDPrimReader *usd_reader = create_fake_reader(archive, prim);

  if (usd_reader == NULL) {
    /* This object is not supported */
    return NULL;
  }
  usd_reader->object(object);
  usd_reader->incref();

  return reinterpret_cast<CacheReader *>(usd_reader);
}

void USDCacheReader_free(CacheReader *reader)
{
  USDPrimReader *usd_reader = reinterpret_cast<USDPrimReader *>(reader);
  usd_reader->decref();

  if (usd_reader->refcount() == 0) {
    delete usd_reader;
  }
}

USDStageHandle *USD_create_handle(struct Main *bmain, const char *filename, ListBase *object_paths)
{
  USDStageReader *archive = new USDStageReader(bmain, filename);

  if (!archive->valid()) {
    delete archive;
    return NULL;
  }

  if (object_paths) {
    gather_objects_paths(archive->stage()->GetPseudoRoot(), object_paths);
  }

  return handle_from_archive(archive);
}

void USD_free_handle(USDStageHandle *handle)
{
  USDStageReader *archive = archive_from_handle(handle);
  delete archive;
}

void USD_get_transform(struct CacheReader *reader,
                       float r_mat_world[4][4],
                       float time,
                       float scale)
{
  if (!reader) {
    return;
  }
  USDXformReader *usd_reader = reinterpret_cast<USDXformReader *>(reader);

  bool is_constant = false;

  /* Convert from the local matrix we obtain from USD to world coordinates
   * for Blender. This conversion is done here rather than by Blender due to
   * work around the non-standard interpretation of CONSTRAINT_SPACE_LOCAL in
   * BKE_constraint_mat_convertspace(). */
  Object *object = usd_reader->object();
  if (object->parent == nullptr) {
    /* No parent, so local space is the same as world space. */
    usd_reader->read_matrix(r_mat_world, time, scale, is_constant);
    return;
  }

  float mat_parent[4][4];
  BKE_object_get_parent_matrix(object, object->parent, mat_parent);

  float mat_local[4][4];
  usd_reader->read_matrix(mat_local, time, scale, is_constant);
  mul_m4_m4m4(r_mat_world, mat_parent, object->parentinv);
  mul_m4_m4m4(r_mat_world, r_mat_world, mat_local);
}
