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
 * Define a basic runtime node type.
 * A custom RNA struct is declared for the type.
 * The node type is not registered here.
 */
void node_make_runtime_type(struct bNodeType *ntype,
                            const char *idname,
                            const char *ui_name,
                            const char *ui_description,
                            int ui_icon,
                            short node_class,
                            const StructRNA *rna_base);

/**
 * Free runtime type information of the node type.
 * The node type is not unregistered here.
 */
void node_free_runtime_type(struct bNodeType *ntype);

#ifdef __cplusplus
}
#endif
