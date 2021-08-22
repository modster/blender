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
 * \ingroup nodes
 */

#pragma once

#include <type_traits>

#include "BKE_node.h"

#include "RNA_access.h"
#include "RNA_types.h"

#include "UI_resources.h"

/** \file
 * \ingroup fn
 *
 * Utility functions for registering node types at runtime.
 *
 * Defining nodes here does not require compile-time DNA (makesdna) or RNA (makesrna).
 * Nodes can use ID properties and runtime RNA definition.
 *
 * Node types can be registered using a C++ class with static fields and functions.
 * Functions are plain C callbacks, not actual class methods.
 * The register function detects missing optional fields and falls back on default values.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Can this node type be added to a node tree?
 * \param r_disabled_hint: Optional hint to display in the UI when the poll fails.
 *                         The callback can set this to a static string without having to
 *                         null-check it (or without setting it to null if it's not used).
 *                         The caller must pass a valid `const char **` and null-initialize it
 *                         when it's not just a dummy, that is, if it actually wants to access
 *                         the returned disabled-hint (null-check needed!).
 */
typedef bool (*NodePollCb)(bNodeType *ntype, bNodeTree *ntree, const char **r_disabled_hint);
/** Can this node be added to a node tree?
 * \param r_disabled_hint: See `poll()`.
 */
typedef bool (*NodeInstancePollCb)(bNode *node, bNodeTree *ntree, const char **r_disabled_hint);
/** Initialize a new node instance of this type after creation. */
typedef void (*NodeInitCb)(struct bNodeTree *ntree, struct bNode *node);
/** Free the node instance. */
typedef void (*NodeFreeCb)(struct bNode *node);
/** Make a copy of the node instance. */
typedef void (*NodeCopyCb)(struct bNodeTree *dest_ntree,
                           struct bNode *dest_node,
                           const struct bNode *src_node);
/* Called after a node socket has been linked. */
typedef void (*NodeInsertLinkCb)(struct bNodeTree *ntree,
                                 struct bNode *node,
                                 struct bNodeLink *link);
/* Update the internal links list, for muting and disconnect operators. */
typedef void (*NodeUpdateInternalLinksCb)(struct bNodeTree *, struct bNode *node);
/* Called when the node is updated in the editor. */
typedef void (*NodeUpdateCb)(struct bNodeTree *ntree, struct bNode *node);
/* Check and update if internal ID data has changed. */
typedef void (*NodeGroupUpdateCb)(struct bNodeTree *ntree, struct bNode *node);
/**
 * Optional custom label function for the node header.
 * \note Used as a fallback when #bNode.label isn't set.
 */
typedef void (*NodeLabelCb)(struct bNodeTree *ntree, struct bNode *node, char *label, int maxlen);
/* Draw the option buttons on the node */
typedef void (*NodeDrawButtonsCb)(struct uiLayout *layout,
                                  struct bContext *C,
                                  struct PointerRNA *ptr);
/* Additional parameters in the side panel */
typedef void (*NodeDrawButtonsExCb)(struct uiLayout *layout,
                                    struct bContext *C,
                                    struct PointerRNA *ptr);
/* Additional drawing on backdrop */
typedef void (*NodeDrawBackdropCb)(
    struct SpaceNode *snode, struct ImBuf *backdrop, struct bNode *node, int x, int y);

/**
 * Define a full node type with all possible callbacks.
 * The node type is not registered here.
 */
void node_make_runtime_type_ex(struct bNodeType *ntype,
                               const char *idname,
                               const char *ui_name,
                               const char *ui_description,
                               int ui_icon,
                               short node_class,
                               const StructRNA *rna_base,
                               NodePollCb poll_cb,
                               NodeInstancePollCb instance_poll_cb,
                               NodeInitCb init_cb,
                               NodeFreeCb free_cb,
                               NodeCopyCb copy_cb,
                               NodeInsertLinkCb insert_link_cb,
                               NodeUpdateInternalLinksCb update_internal_links_cb,
                               NodeUpdateCb update_cb,
                               NodeGroupUpdateCb group_update_cb,
                               NodeLabelCb label_cb,
                               NodeDrawButtonsCb draw_buttons_cb,
                               NodeDrawButtonsExCb draw_buttons_ex_cb,
                               NodeDrawBackdropCb draw_backdrop_cb,
                               eNodeSizePreset size_preset);

/**
 * Free runtime type information of the node type.
 * The node type is not unregistered here.
 */
void node_free_runtime_type(struct bNodeType *ntype);

#ifdef __cplusplus
}
#endif

namespace blender::nodes {

namespace detail {

/** Utility macro expanding to a field of the node class.
 * Used for consistency with more complex optional field macros below.
 */
#define DECL_NODE_FIELD_REQUIRED(FIELD_NAME) \
  template<typename T> static constexpr auto node_type_get__##FIELD_NAME() \
  { \
    return T::FIELD_NAME; \
  }
#define DECL_NODE_FUNC_REQUIRED(FUNC_NAME) \
  template<typename T> static constexpr auto node_type_get__##FUNC_NAME() \
  { \
    return &T::FUNC_NAME; \
  }

/**
 * Utility macro for selecting a static field from the node class,
 * or a default value if the field is not defined.
 *
 * The int/long argument is used to prioritize the T::FIELD_NAME implementation
 * if the field is defined: the 0 literal is preferably interpreted as int.
 * Only if the field does not exist, i.e. decltype(T::FIELD_NAME) is invalid,
 * will the default value implementation be used (SFINAE).
 *
 * TODO C++20 introduces "concepts" and the "requires" keyword,
 * which are more elegant ways to handle missing fields on template arguments.
 */
#define DECL_NODE_FIELD_OPTIONAL(FIELD_NAME, DEFAULT_VALUE) \
  template<typename T> \
  static constexpr auto node_type_impl__##FIELD_NAME(int)->decltype(T::FIELD_NAME) \
  { \
    return T::FIELD_NAME; \
  } \
  template<typename T> \
  static constexpr auto node_type_impl__##FIELD_NAME(long)->decltype(DEFAULT_VALUE) \
  { \
    return DEFAULT_VALUE; \
  } \
  template<typename T> static constexpr auto node_type_get__##FIELD_NAME() \
  { \
    return node_type_impl__##FIELD_NAME<T>(0); \
  }
#define DECL_NODE_FUNC_OPTIONAL(FUNC_NAME) \
  template<typename T> \
  static constexpr auto node_type_impl__##FUNC_NAME(int)->decltype(&T::FUNC_NAME) \
  { \
    return &T::FUNC_NAME; \
  } \
  template<typename T> static constexpr auto node_type_impl__##FUNC_NAME(long)->decltype(nullptr) \
  { \
    return nullptr; \
  } \
  template<typename T> static constexpr auto node_type_get__##FUNC_NAME() \
  { \
    return node_type_impl__##FUNC_NAME<T>(0); \
  }

/* Required and optional fields and callbacks expected in node type classes. */
DECL_NODE_FIELD_REQUIRED(idname)
DECL_NODE_FIELD_REQUIRED(ui_name)
DECL_NODE_FIELD_OPTIONAL(ui_description, "")
DECL_NODE_FIELD_OPTIONAL(ui_icon, ICON_NONE)
DECL_NODE_FIELD_REQUIRED(node_class)
DECL_NODE_FIELD_OPTIONAL(rna_base, &RNA_Node)
DECL_NODE_FUNC_OPTIONAL(poll)
DECL_NODE_FUNC_OPTIONAL(instance_poll)
DECL_NODE_FUNC_OPTIONAL(init)
DECL_NODE_FUNC_OPTIONAL(free)
DECL_NODE_FUNC_OPTIONAL(copy)
DECL_NODE_FUNC_OPTIONAL(insert_link)
DECL_NODE_FUNC_OPTIONAL(update_internal_links)
DECL_NODE_FUNC_OPTIONAL(update)
DECL_NODE_FUNC_OPTIONAL(group_update)
DECL_NODE_FUNC_OPTIONAL(label)
DECL_NODE_FUNC_OPTIONAL(draw_buttons)
DECL_NODE_FUNC_OPTIONAL(draw_buttons_ex)
DECL_NODE_FUNC_OPTIONAL(draw_backdrop)
DECL_NODE_FIELD_OPTIONAL(size_preset, NODE_SIZE_DEFAULT)

DECL_NODE_FUNC_OPTIONAL(define_rna)

}  // namespace detail

/* Template for runtime node definition. */
template<typename T> struct NodeDefinition {
  inline static bNodeType typeinfo_;

  /* Registers a node type using static fields and callbacks of the template argument. */
  static void register_type()
  {
    ::node_make_runtime_type_ex(&typeinfo_,
                                detail::node_type_get__idname<T>(),
                                detail::node_type_get__ui_name<T>(),
                                detail::node_type_get__ui_description<T>(),
                                detail::node_type_get__ui_icon<T>(),
                                detail::node_type_get__node_class<T>(),
                                detail::node_type_get__rna_base<T>(),
                                detail::node_type_get__poll<T>(),
                                detail::node_type_get__instance_poll<T>(),
                                detail::node_type_get__init<T>(),
                                detail::node_type_get__free<T>(),
                                detail::node_type_get__copy<T>(),
                                detail::node_type_get__insert_link<T>(),
                                detail::node_type_get__update_internal_links<T>(),
                                detail::node_type_get__update<T>(),
                                detail::node_type_get__group_update<T>(),
                                detail::node_type_get__label<T>(),
                                detail::node_type_get__draw_buttons<T>(),
                                detail::node_type_get__draw_buttons_ex<T>(),
                                detail::node_type_get__draw_backdrop<T>(),
                                detail::node_type_get__size_preset<T>());

    /* Call optional RNA setup function for the type. */
    if (void (*define_rna_cb)(struct StructRNA *) = detail::node_type_get__define_rna<T>()) {
      define_rna_cb(typeinfo_.rna_ext.srna);
    }

    nodeRegisterType(&typeinfo_);
  }

  static void unregister_type()
  {
    ::node_free_runtime_type(&typeinfo_);

    nodeUnregisterType(&typeinfo_);
  };

};

}  // namespace blender::nodes
