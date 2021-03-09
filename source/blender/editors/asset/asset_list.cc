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
 * \ingroup edasset
 *
 * Abstractions to manage runtime asset lists with a global cache for multiple UI elements to
 * access.
 * Internally this uses the #FileList API and structures from `filelist.c`. This is just because it
 * contains most necessary logic already and there's not much time for a more long-term solution.
 */

#include <optional>

#include "BLI_function_ref.hh"
#include "BLI_hash.hh"
#include "BLI_map.hh"

#include "DNA_asset_types.h"
#include "DNA_space_types.h"

#include "BKE_preferences.h"

#include "ED_asset.h"
#include "ED_fileselect.h"

/* XXX uses private header of file-space. */
#include "../space_file/filelist.h"

using namespace blender;

/**
 * Wrapper to add logic to the AssetLibraryReference DNA struct.
 */
struct AssetLibraryReferenceWrapper {
  const AssetLibraryReference &reference_;

  AssetLibraryReferenceWrapper(const AssetLibraryReference &reference) : reference_(reference)
  {
  }
  ~AssetLibraryReferenceWrapper() = default;

  friend bool operator==(const AssetLibraryReferenceWrapper &a,
                         const AssetLibraryReferenceWrapper &b)
  {
    return (a.reference_.type == b.reference_.type) &&
                   (a.reference_.type == ASSET_LIBRARY_CUSTOM) ?
               (a.reference_.custom_library_index == b.reference_.custom_library_index) :
               true;
  }

  uint64_t hash() const
  {
    uint64_t hash1 = DefaultHash<decltype(reference_.type)>{}(reference_.type);
    if (reference_.type != ASSET_LIBRARY_CUSTOM) {
      return hash1;
    }

    uint64_t hash2 = DefaultHash<decltype(reference_.custom_library_index)>{}(
        reference_.custom_library_index);
    return hash1 ^ (hash2 * 33); /* Copied from DefaultHash for std::pair. */
  }
};

/* -------------------------------------------------------------------- */
/** \name Asset list API
 *
 *  Internally re-uses #FileList from the File Browser. It does all the heavy lifting already.
 * \{ */

#include "BKE_context.h"
#include "WM_api.h"
#include "WM_types.h"

class AssetList {
  static void filelist_free_wrapper(FileList *list)
  {
    filelist_free(list);
    MEM_freeN(list);
  }

  using FileList_Ptr = std::unique_ptr<FileList, decltype(&filelist_free_wrapper)>;
  FileList_Ptr filelist_;
  AssetLibraryReference library_ref_;

  struct wmTimer *previews_timer = nullptr;

 public:
  AssetList() = delete;
  AssetList(eFileSelectType filesel_type, const AssetLibraryReference &asset_library_ref)
      : filelist_(filelist_new(filesel_type), filelist_free_wrapper),
        library_ref_(asset_library_ref)
  {
  }
  AssetList(AssetList &&other)
      : filelist_(std::move(other.filelist_)), library_ref_(other.library_ref_)
  {
  }
  AssetList(const AssetList &) = delete;
  ~AssetList() = default;

  void setup(const AssetFilterSettings *filter_settings = nullptr)
  {
    FileList *files = filelist_.get();

    /* TODO there should only be one. */
    FileSelectAssetLibraryUID file_asset_lib_ref;
    file_asset_lib_ref.type = library_ref_.type;
    file_asset_lib_ref.custom_library_index = library_ref_.custom_library_index;

    bUserAssetLibrary *user_library = NULL;

    /* Ensure valid repository, or fall-back to local one. */
    if (library_ref_.type == ASSET_LIBRARY_CUSTOM) {
      BLI_assert(library_ref_.custom_library_index >= 0);

      user_library = BKE_preferences_asset_library_find_from_index(
          &U, library_ref_.custom_library_index);
    }

    char path[FILE_MAXDIR] = "";
    if (user_library) {
      BLI_strncpy(path, user_library->path, sizeof(path));
      filelist_setdir(files, path);
    }
    else {
      filelist_setdir(files, path);
    }

    /* Relevant bits from file_refresh(). */
    /* TODO pass options properly. */
    filelist_setrecursion(files, 1);
    filelist_setsorting(files, FILE_SORT_ALPHA, false);
    filelist_setlibrary(files, &file_asset_lib_ref);
    /* TODO different filtering settings require the list to be reread. That's a no-go for when we
     * want to allow showing the same asset library with different filter settings (as in,
     * different ID types). The filelist needs to be made smarter somehow, maybe goes together with
     * the plan to separate the view (preview caching, filtering, etc. ) from the data. */
    filelist_setfilter_options(
        files,
        true,
        true,
        true, /* Just always hide parent, prefer to not add an extra user option for this. */
        FILE_TYPE_BLENDERLIB,
        filter_settings ? filter_settings->id_types : FILTER_ID_ALL,
        true,
        "",
        "");
  }

  void fetch(const bContext &C)
  {
    FileList *files = filelist_.get();

    if (filelist_needs_reading(files)) {
      if (!filelist_pending(files)) {
        filelist_readjob_start(files, &C);
      }
    }
    filelist_sort(files);
    filelist_filter(files);
  }

  void iterate(AssetListIterFn fn)
  {
    FileList *files = filelist_.get();
    int numfiles = filelist_files_ensure(files);

    for (int i = 0; i < numfiles; i++) {
      FileDirEntry *file = filelist_file(files, i);
      if (!fn(*file)) {
        break;
      }
    }
  }

  void ensurePreviewsJob(bContext *C)
  {
    FileList *files = filelist_.get();
    int numfiles = filelist_files_ensure(files);

    filelist_cache_previews_set(files, true);
    filelist_file_cache_slidingwindow_set(files, 256);
    /* TODO fetch all previews for now. */
    filelist_file_cache_block(files, numfiles / 2);
    filelist_cache_previews_update(files);

    /* TODO Copied from file_draw_list() */
    {
      const bool previews_running = filelist_cache_previews_running(files);
      if (previews_running && !previews_timer) {
        /* TODO NC_WINDOW_to force full redraw.  */
        previews_timer = WM_event_add_timer_notifier(
            CTX_wm_manager(C), CTX_wm_window(C), NC_WINDOW, 0.01);
      }
      if (!previews_running && previews_timer) {
        /* Preview is not running, no need to keep generating update events! */
        WM_event_remove_timer_notifier(CTX_wm_manager(C), CTX_wm_window(C), previews_timer);
        previews_timer = NULL;
      }
    }
  }

  blender::StringRef filepath()
  {
    return filelist_dir(filelist_.get());
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Runtime asset list cache
 * \{ */

/**
 * Class managing a global asset list map, each entry being a list for a specific asset library.
 */
class AssetListStorage {
  using AssetListMap = Map<AssetLibraryReferenceWrapper, AssetList>;
  static AssetListMap global_storage_;

 public:
  static void fetch_library(const AssetLibraryReference &library_reference,
                            const bContext &C,
                            const AssetFilterSettings *filter_settings = nullptr)
  {
    std::optional filesel_type = asset_library_reference_to_fileselect_type(library_reference);
    if (!filesel_type) {
      return;
    }

    std::tuple list_create_info = ensure_list_storage(library_reference, *filesel_type);
    AssetList &list = std::get<0>(list_create_info);
    const bool is_new = std::get<1>(list_create_info);
    if (is_new) {
      list.setup(filter_settings);
      list.fetch(C);
    }
  }

  static void destruct()
  {
    global_storage_.~AssetListMap();
  }

  static AssetList *lookup_list(const AssetLibraryReference &library_ref)
  {
    return global_storage_.lookup_ptr(library_ref);
  }

 private:
  /* Private constructor. Can't instantiate this. */
  AssetListStorage() = default;

  static std::optional<eFileSelectType> asset_library_reference_to_fileselect_type(
      const AssetLibraryReference &library_reference)
  {
    switch (library_reference.type) {
      case ASSET_LIBRARY_CUSTOM:
        return FILE_LOADLIB;
      case ASSET_LIBRARY_LOCAL:
        return FILE_MAIN_ASSET;
    }

    return std::nullopt;
  }

  using is_new_t = bool;
  static std::tuple<AssetList &, is_new_t> ensure_list_storage(
      const AssetLibraryReference &library_reference, eFileSelectType filesel_type)
  {
    if (AssetList *list = global_storage_.lookup_ptr(library_reference)) {
      return {*list, false};
    }
    global_storage_.add(library_reference, AssetList(filesel_type, library_reference));
    return {global_storage_.lookup(library_reference), true};
  }
};

AssetListStorage::AssetListMap AssetListStorage::global_storage_{};

/** \} */

/* -------------------------------------------------------------------- */
/** \name C-API
 * \{ */

/**
 * Invoke asset list reading, potentially in a parallel job. Won't wait until the job is done,
 * and may return earlier.
 */
void ED_assetlist_fetch(const AssetLibraryReference *library_reference,
                        const AssetFilterSettings *filter_settings,
                        const bContext *C)
{
  AssetListStorage::fetch_library(*library_reference, *C, filter_settings);
}

void ED_assetlist_ensure_previews_job(const AssetLibraryReference *library_reference, bContext *C)
{

  AssetList *list = AssetListStorage::lookup_list(*library_reference);
  if (list) {
    list->ensurePreviewsJob(C);
  }
}

void ED_assetlist_storage_exit()
{
  AssetListStorage::destruct();
}

/* TODO expose AssetList with an iterator? */
void ED_assetlist_iterate(const AssetLibraryReference *library_reference, AssetListIterFn fn)
{
  AssetList *list = AssetListStorage::lookup_list(*library_reference);
  if (list) {
    list->iterate(fn);
  }
}

ImBuf *ED_assetlist_asset_image_get(const FileDirEntry *file)
{
  ImBuf *imbuf = filelist_file_getimage(file);
  if (imbuf) {
    return imbuf;
  }

  return filelist_geticon_image_ex(file);
}

const char *ED_assetlist_library_path(const AssetLibraryReference *library_reference)
{
  AssetList *list = AssetListStorage::lookup_list(*library_reference);
  if (list) {
    return list->filepath().data();
  }
  return nullptr;
}

/** \} */
