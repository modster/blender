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

#include "BKE_screen.h"

#include "BLI_function_ref.hh"
#include "BLI_hash.hh"
#include "BLI_map.hh"

#include "DNA_asset_types.h"
#include "DNA_space_types.h"

#include "BKE_preferences.h"

#include "ED_asset.h"
#include "ED_fileselect.h"
#include "ED_screen.h"

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
  ~AssetList()
  {
    /* Destructs the owned pointer. */
    filelist_ = nullptr;
  }

  void setup(const AssetFilterSettings *filter_settings = nullptr)
  {
    FileList *files = filelist_.get();

    /* TODO there should only be one (FileSelectAssetLibraryUID vs. AssetLibraryReference). */
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
        filter_settings != nullptr,
        true,
        true, /* Just always hide parent, prefer to not add an extra user option for this. */
        FILE_TYPE_BLENDERLIB,
        filter_settings ? filter_settings->id_types : FILTER_ID_ALL,
        true,
        "",
        "");

    char path[FILE_MAXDIR] = "";
    if (user_library) {
      BLI_strncpy(path, user_library->path, sizeof(path));
      filelist_setdir(files, path);
    }
    else {
      filelist_setdir(files, path);
    }
  }

  void fetch(const bContext &C)
  {
    FileList *files = filelist_.get();

    if (filelist_needs_force_reset(files)) {
      filelist_readjob_stop(CTX_wm_manager(&C), CTX_data_scene(&C));
      filelist_clear(files);
    }

    if (filelist_needs_reading(files)) {
      if (!filelist_pending(files)) {
        filelist_readjob_start(files, NC_ASSET | ND_ASSET_LIST_READING, &C);
      }
    }
    filelist_sort(files);
    filelist_filter(files);
  }

  bool needsRefetch() const
  {
    return filelist_needs_force_reset(filelist_.get());
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
        previews_timer = WM_event_add_timer_notifier(
            CTX_wm_manager(C), CTX_wm_window(C), NC_ASSET | ND_ASSET_LIST_PREVIEW, 0.01);
      }
      if (!previews_running && previews_timer) {
        /* Preview is not running, no need to keep generating update events! */
        WM_event_remove_timer_notifier(CTX_wm_manager(C), CTX_wm_window(C), previews_timer);
        previews_timer = NULL;
      }
    }
  }

  /**
   * \return True if the asset-list needs a UI redraw.
   */
  bool listen(const wmNotifier &notifier) const
  {
    switch (notifier.category) {
      case NC_ASSET:
        if (ELEM(notifier.data, ND_ASSET_LIST_READING, ND_ASSET_LIST_PREVIEW)) {
          return true;
        }
        if (ELEM(notifier.action, NA_ADDED, NA_REMOVED)) {
          return true;
        }
        break;
    }

    return false;
  }

  void tagMainDataDirty() const
  {
    FileList *files = filelist_.get();

    if (filelist_needs_reset_on_main_changes(files)) {
      /* Full refresh of the file list if local asset data was changed. Refreshing this view
       * is cheap and users expect this to be updated immediately. */
      filelist_tag_force_reset(files);
    }
  }

  void remapID(ID * /*id_old*/, ID * /*id_new*/) const
  {
    /* Trigger full refetch  of the file list if main data was changed, don't even attempt remap
     * pointers. We could give file list types a id-remap callback, but it's probably not worth it.
     * Refreshing local file lists is relatively cheap. */
    tagMainDataDirty();
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
    if (is_new || list.needsRefetch()) {
      list.setup(filter_settings);
      list.fetch(C);
    }
  }

  static void destruct()
  {
    global_storage().~AssetListMap();
  }

  static AssetList *lookup_list(const AssetLibraryReference &library_ref)
  {
    return global_storage().lookup_ptr(library_ref);
  }

  static void tagMainDataDirty()
  {
    for (AssetList &list : global_storage().values()) {
      list.tagMainDataDirty();
    }
  }

  static void remapID(ID *id_new, ID *id_old)
  {
    for (AssetList &list : global_storage().values()) {
      list.remapID(id_new, id_old);
    }
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
    AssetListMap &storage = global_storage();

    if (AssetList *list = storage.lookup_ptr(library_reference)) {
      return {*list, false};
    }
    storage.add(library_reference, AssetList(filesel_type, library_reference));
    return {storage.lookup(library_reference), true};
  }

  /**
   * Wrapper for Construct on First Use idiom, to avoid the Static Initialization Fiasco.
   */
  static AssetListMap &global_storage()
  {
    static AssetListMap global_storage_;
    return global_storage_;
  }
};

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

/**
 * \return True if the region needs a UI redraw.
 */
bool ED_assetlist_listen(const AssetLibraryReference *library_reference,
                         const wmNotifier *notifier)
{
  AssetList *list = AssetListStorage::lookup_list(*library_reference);
  if (list) {
    return list->listen(*notifier);
  }
  return false;
}

/**
 * Tag all asset lists in the storage that show main data as needing an update (refetch).
 *
 * This only tags the data. If the asset list is visible on screen, the space is still responsible
 * for ensuring the necessary redraw. It can use #ED_assetlist_listen() to check if the asset-list
 * needs a redraw for a given notifier.
 */
void ED_assetlist_storage_tag_main_data_dirty()
{
  AssetListStorage::tagMainDataDirty();
}

/**
 * Remapping of ID pointers within the asset lists. Typically called when an ID is deleted to clear
 * all references to it (\a id_new is null then).
 */
void ED_assetlist_storage_id_remap(ID *id_old, ID *id_new)
{
  AssetListStorage::remapID(id_old, id_new);
}

/**
 * Can't wait for static deallocation to run. There's nested data allocated with our guarded
 * allocator, it will complain about unfreed memory on exit.
 */
void ED_assetlist_storage_exit()
{
  AssetListStorage::destruct();
}

/** \} */
