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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

extern struct bNodeTreeType *ntreeType_Function;

void register_node_tree_type_function(void);

void register_node_type_function_group(void);

void register_node_type_fn_boolean_math(void);
void register_node_type_fn_float_compare(void);
void register_node_type_fn_input_string(void);
void register_node_type_fn_input_vector(void);
void register_node_type_fn_random_float(void);

#ifdef __cplusplus
}
#endif
