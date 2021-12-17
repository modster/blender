
#include "BKE_lib_remap.h"

#include "MEM_guardedalloc.h"

#include "BLI_map.hh"

namespace blender::bke::id::remapper {
struct IDRemapper {
 private:
  Map<ID *, ID *> mappings;

 public:
  void add(ID *old_id, ID *new_id)
  {
    BLI_assert(old_id != nullptr);
    mappings.add_as(old_id, new_id);
  }

  bool apply(ID **id_ptr_ptr) const
  {
    BLI_assert(id_ptr_ptr != nullptr);
    if (*id_ptr_ptr == nullptr) {
      return false;
    }

    if (!mappings.contains(*id_ptr_ptr)) {
      return false;
    }

    *id_ptr_ptr = mappings.lookup(*id_ptr_ptr);
    return true;
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

bool BKE_id_remapper_apply(const IDRemapper *id_remapper, ID **id_ptr_ptr)
{
  const blender::bke::id::remapper::IDRemapper *remapper = unwrap_const(id_remapper);
  return remapper->apply(id_ptr_ptr);
}
}