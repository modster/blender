
#include "DNA_ID.h"

#include "BKE_idtype.h"
#include "BKE_lib_id.h"
#include "BKE_lib_remap.h"

#include "MEM_guardedalloc.h"

#include "BLI_map.hh"

using IDTypeFilter = uint64_t;

namespace blender::bke::id::remapper {
struct IDRemapper {
 private:
  Map<ID *, ID *> mappings;
  IDTypeFilter source_types = 0;

 public:
  void add(ID *old_id, ID *new_id)
  {
    BLI_assert(old_id != nullptr);
    mappings.add_as(old_id, new_id);
    source_types |= BKE_idtype_idcode_to_idfilter(GS(old_id->name));
  }

  bool contains_mappings_for_any(IDTypeFilter filter) const
  {
    return (source_types & filter) != 0;
  }

  IDRemapperApplyResult apply(ID **id_ptr_ptr, IDRemapperApplyOptions options) const
  {
    BLI_assert(id_ptr_ptr != nullptr);
    if (*id_ptr_ptr == nullptr) {
      return ID_REMAP_SOURCE_NOT_MAPPABLE;
    }

    if (!mappings.contains(*id_ptr_ptr)) {
      return ID_REMAP_SOURCE_UNAVAILABLE;
    }

    if (options & ID_REMAP_APPLY_UPDATE_REFCOUNT) {
      id_us_min(*id_ptr_ptr);
    }

    *id_ptr_ptr = mappings.lookup(*id_ptr_ptr);
    if (*id_ptr_ptr == nullptr) {
      return ID_REMAP_SOURCE_UNASSIGNED;
    }

    if (options & ID_REMAP_APPLY_UPDATE_REFCOUNT) {
      id_us_plus(*id_ptr_ptr);
    }

    if (options & ID_REMAP_APPLY_ENSURE_REAL) {
      id_us_ensure_real(*id_ptr_ptr);
    }
    return ID_REMAP_SOURCE_REMAPPED;
  }

  void iter(IDRemapperIterFunction func, void *user_data) const
  {
    for (auto item : mappings.items()) {
      func(item.key, item.value, user_data);
    }
  }
};

}  // namespace blender::bke::id::remapper

extern "C" {
/** \brief wrap CPP IDRemapper to a C handle. */
static IDRemapper *wrap(blender::bke::id::remapper::IDRemapper *remapper)
{
  return static_cast<IDRemapper *>(static_cast<void *>(remapper));
}

/** \brief wrap C handle to a CPP IDRemapper. */
static blender::bke::id::remapper::IDRemapper *unwrap(IDRemapper *remapper)
{
  return static_cast<blender::bke::id::remapper::IDRemapper *>(static_cast<void *>(remapper));
}

/** \brief wrap C handle to a CPP IDRemapper. */
static const blender::bke::id::remapper::IDRemapper *unwrap_const(const IDRemapper *remapper)
{
  return static_cast<const blender::bke::id::remapper::IDRemapper *>(
      static_cast<const void *>(remapper));
}

IDRemapper *BKE_id_remapper_create(void)
{
  blender::bke::id::remapper::IDRemapper *remapper =
      MEM_new<blender::bke::id::remapper::IDRemapper>(__func__);
  return wrap(remapper);
}

void BKE_id_remapper_free(IDRemapper *id_remapper)
{
  blender::bke::id::remapper::IDRemapper *remapper = unwrap(id_remapper);
  MEM_delete<blender::bke::id::remapper::IDRemapper>(remapper);
}

void BKE_id_remapper_add(IDRemapper *id_remapper, ID *old_id, ID *new_id)
{
  blender::bke::id::remapper::IDRemapper *remapper = unwrap(id_remapper);
  remapper->add(old_id, new_id);
}

bool BKE_id_remapper_has_mapping_for(const struct IDRemapper *id_remapper, uint64_t type_filter)
{
  const blender::bke::id::remapper::IDRemapper *remapper = unwrap_const(id_remapper);
  return remapper->contains_mappings_for_any(type_filter);
}

IDRemapperApplyResult BKE_id_remapper_apply(const IDRemapper *id_remapper,
                                            ID **id_ptr_ptr,
                                            const IDRemapperApplyOptions options)
{
  const blender::bke::id::remapper::IDRemapper *remapper = unwrap_const(id_remapper);
  return remapper->apply(id_ptr_ptr, options);
}

void BKE_id_remapper_iter(const struct IDRemapper *id_remapper,
                          IDRemapperIterFunction func,
                          void *user_data)
{
  const blender::bke::id::remapper::IDRemapper *remapper = unwrap_const(id_remapper);
  remapper->iter(func, user_data);
}
}