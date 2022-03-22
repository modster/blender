/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup spoutliner
 */

#pragma once

#include "RNA_types.h"

#include "tree_element.hh"

struct ID;
struct IDOverrideLibraryProperty;

namespace blender::ed::outliner {

struct TreeElementOverridesData {
  ID &id;
  IDOverrideLibraryProperty &override_property;
  PointerRNA &override_rna_ptr;
  PropertyRNA &override_rna_prop;

  bool is_rna_path_valid;
};

class TreeElementOverridesBase final : public AbstractTreeElement {
 public:
  ID &id;

 public:
  TreeElementOverridesBase(TreeElement &legacy_te, ID &id);

  void expand(SpaceOutliner &) const override;
};

/**
 * Common interface for any item nested under #TreeElementOverridesBase. It gives access to the ID
 * owning the override.
 */
class TreeElementOverridesItem : public AbstractTreeElement {
 public:
  ID *getOverrideOwnerID();

 protected:
  using AbstractTreeElement::AbstractTreeElement;
};

class TreeElementOverridesProperty final : public TreeElementOverridesItem {
  [[maybe_unused]] IDOverrideLibraryProperty &override_prop_;

 public:
  PointerRNA override_rna_ptr;
  PropertyRNA &override_rna_prop;

 public:
  TreeElementOverridesProperty(TreeElement &legacy_te, TreeElementOverridesData &override_data);
};

/**
 * If the override is within some collection or pointer property, the collection/pointer gets its
 * own parent item with items inside.
 */
class TreeElementOverrideRNAContainer final : public TreeElementOverridesItem {
 public:
  PointerRNA container_ptr;
  PropertyRNA &container_prop;

 public:
  TreeElementOverrideRNAContainer(TreeElement &legacy_te,
                                  PropertyPointerRNA &container_prop_and_ptr);

  void expand(SpaceOutliner &) const override;
};

class TreeElementOverrideRNACollectionItem final : public TreeElementOverridesItem {
 public:
  PointerRNA item_ptr;

 public:
  TreeElementOverrideRNACollectionItem(TreeElement &legacy_te, const PointerRNA &item_ptr);

  int getIcon() const;
};

}  // namespace blender::ed::outliner
