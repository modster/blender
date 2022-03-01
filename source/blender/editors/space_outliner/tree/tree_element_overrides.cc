/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup spoutliner
 */

#include "BKE_collection.h"
#include "BKE_lib_override.h"

#include "BLI_utildefines.h"

#include "BLI_listbase_wrapper.hh"

#include "BLT_translation.h"

#include "DNA_space_types.h"

#include "RNA_access.h"

#include "../outliner_intern.hh"

#include "tree_element_overrides.hh"

namespace blender::ed::outliner {

TreeElementOverridesBase::TreeElementOverridesBase(TreeElement &legacy_te, ID &id)
    : AbstractTreeElement(legacy_te), id_(id)
{
  BLI_assert(legacy_te.store_elem->type == TSE_LIBRARY_OVERRIDE_BASE);
  if (legacy_te.parent != nullptr &&
      ELEM(legacy_te.parent->store_elem->type, TSE_SOME_ID, TSE_LAYER_COLLECTION))

  {
    legacy_te.name = IFACE_("Library Overrides");
  }
  else {
    legacy_te.name = id.name + 2;
  }
}

static void expand_from_rna_path(SpaceOutliner &space_outliner,
                                 TreeElement &path_root_te,
                                 TreeElementOverridesData &override_data,
                                 int *index)
{
  PointerRNA idpoin;
  RNA_id_pointer_create(&override_data.id, &idpoin);

  TreeElement *parent_to_expand = &path_root_te;

  ListBase path_elems{nullptr};
  RNA_path_resolve_elements(&idpoin, override_data.override_property.rna_path, &path_elems);

  /* Iterate the properties represented by the path. */
  LISTBASE_FOREACH (PropertyElemRNA *, prop_ptr_from_path, &path_elems) {
    /* Create containers for all items that are not leafs (i.e. that are not simple properties, but
     * may contain child properties). */
    if (prop_ptr_from_path->next) {
      PropertyPointerRNA property_and_ptr = {prop_ptr_from_path->ptr, prop_ptr_from_path->prop};
      TreeElement *container_te = outliner_add_element(&space_outliner,
                                                       &parent_to_expand->subtree,
                                                       &property_and_ptr,
                                                       parent_to_expand,
                                                       TSE_LIBRARY_OVERRIDE_RNA_CONTAINER,
                                                       *index++);

      parent_to_expand = container_te;
      /* Iterate over the children the container item expanded, and continue building the path for
       * the item that matches the current path segment. */
      LISTBASE_FOREACH (TreeElement *, container_item_te, &container_te->subtree) {
        if (auto *col_item_te = tree_element_cast<TreeElementOverrideRNACollectionItem>(
                container_item_te)) {
          /* Does the collection item RNA pointer match the RNA pointer of the next property in the
           * path? */
          if (col_item_te->item_ptr.data == prop_ptr_from_path->next->ptr.data) {
            parent_to_expand = &col_item_te->getLegacyElement();
          }
        }
      }

      continue;
    }

    /* The actually overridden property. Must be a "leaf" property (end of the path). */
    BLI_assert(prop_ptr_from_path->next == nullptr);
    /* The actual override. */
    outliner_add_element(&space_outliner,
                         &parent_to_expand->subtree,
                         &override_data,
                         parent_to_expand,
                         TSE_LIBRARY_OVERRIDE,
                         *index++);
  }

  BLI_freelistN(&path_elems);
}

void TreeElementOverridesBase::expand(SpaceOutliner &space_outliner) const
{
  BLI_assert(id_.override_library != nullptr);

  const bool show_system_overrides = (SUPPORT_FILTER_OUTLINER(&space_outliner) &&
                                      (space_outliner.filter & SO_FILTER_SHOW_SYSTEM_OVERRIDES) !=
                                          0);
  PointerRNA idpoin;
  RNA_id_pointer_create(&id_, &idpoin);

  PointerRNA override_rna_ptr;
  PropertyRNA *override_rna_prop;
  int index = 0;

  for (IDOverrideLibraryProperty *override_prop :
       ListBaseWrapper<IDOverrideLibraryProperty>(id_.override_library->properties)) {
    const bool is_rna_path_valid = BKE_lib_override_rna_property_find(
        &idpoin, override_prop, &override_rna_ptr, &override_rna_prop);
    if (is_rna_path_valid && !show_system_overrides &&
        ELEM(override_prop->rna_prop_type, PROP_POINTER, PROP_COLLECTION) &&
        RNA_struct_is_ID(RNA_property_pointer_type(&override_rna_ptr, override_rna_prop))) {
      bool do_continue = true;
      for (IDOverrideLibraryPropertyOperation *override_prop_op :
           ListBaseWrapper<IDOverrideLibraryPropertyOperation>(override_prop->operations)) {
        if ((override_prop_op->flag & IDOVERRIDE_LIBRARY_FLAG_IDPOINTER_MATCH_REFERENCE) == 0) {
          do_continue = false;
          break;
        }
      }

      if (do_continue) {
        continue;
      }
    }

    TreeElementOverridesData override_data = {
        id_, *override_prop, override_rna_ptr, *override_rna_prop, is_rna_path_valid};
    expand_from_rna_path(space_outliner, legacy_te_, override_data, &index);
  }
}

TreeElementOverridesProperty::TreeElementOverridesProperty(TreeElement &legacy_te,
                                                           TreeElementOverridesData &override_data)
    : AbstractTreeElement(legacy_te),
      override_prop_(override_data.override_property),
      override_rna_ptr(override_data.override_rna_ptr),
      override_rna_prop(override_data.override_rna_prop)
{
  BLI_assert(legacy_te.store_elem->type == TSE_LIBRARY_OVERRIDE);

  legacy_te.name = RNA_property_identifier(&override_rna_prop);
  /* Abusing this for now, better way to do it is also pending current refactor of the whole tree
   * code to use C++. */
  legacy_te.directdata = POINTER_FROM_UINT(override_data.is_rna_path_valid);
}

TreeElementOverrideRNAContainer::TreeElementOverrideRNAContainer(
    TreeElement &legacy_te, PropertyPointerRNA &container_prop_and_ptr)
    : AbstractTreeElement(legacy_te),
      container_ptr(container_prop_and_ptr.ptr),
      container_prop(*container_prop_and_ptr.prop)
{
  BLI_assert(legacy_te.store_elem->type == TSE_LIBRARY_OVERRIDE_RNA_CONTAINER);
  legacy_te.name = RNA_property_ui_name(&container_prop);
}

void TreeElementOverrideRNAContainer::expand(SpaceOutliner &space_outliner) const
{
  if (RNA_property_type(&container_prop) != PROP_COLLECTION) {
    /* Only expand RNA collections. For them the exact item order may matter (e.g. for modifiers),
     * so display them all to provide full context. */
    return;
  }

  int index = 0;
  /* Non-const copy. */
  PointerRNA ptr = container_ptr;
  RNA_PROP_BEGIN (&ptr, itemptr, &container_prop) {
    outliner_add_element(&space_outliner,
                         &legacy_te_.subtree,
                         &itemptr,
                         &legacy_te_,
                         TSE_LIBRARY_OVERRIDE_RNA_COLLECTION_ITEM,
                         index++);
  }
  RNA_PROP_END;
}

TreeElementOverrideRNACollectionItem::TreeElementOverrideRNACollectionItem(
    TreeElement &legacy_te, const PointerRNA &item_ptr)
    : AbstractTreeElement(legacy_te), item_ptr(item_ptr)
{
  BLI_assert(legacy_te.store_elem->type == TSE_LIBRARY_OVERRIDE_RNA_COLLECTION_ITEM);
  /* Non-const copy. */
  PointerRNA ptr = item_ptr;
  PropertyRNA *name_prop = RNA_struct_name_property(item_ptr.type);
  legacy_te.name = RNA_property_string_get_alloc(&ptr, name_prop, nullptr, 0, nullptr);
  legacy_te.flag |= TE_FREE_NAME;
}

int TreeElementOverrideRNACollectionItem::getIcon() const
{
  return RNA_struct_ui_icon(item_ptr.type);
}

}  // namespace blender::ed::outliner
