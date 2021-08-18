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

#include "BLI_math_rotation.h"
#include "BLI_task.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_function_util.hh"

using blender::float3;

static bNodeSocketTemplate fn_node_align_rotation_to_vector_in[] = {
    {SOCK_VECTOR, N_("Rotation"), 1.0, 0.0, 0.0, 0.0, -FLT_MAX, FLT_MAX, PROP_EULER},
    {SOCK_FLOAT, N_("Factor"), 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, PROP_FACTOR},
    {SOCK_VECTOR, N_("Vector"), 0.0, 0.0, 1.0, 0.0, -FLT_MAX, FLT_MAX},
    {-1, ""},
};

static bNodeSocketTemplate fn_node_align_rotation_to_vector_out[] = {
    {SOCK_VECTOR, N_("Rotation")},
    {-1, ""},
};

static void fn_node_align_rotation_to_vector_layout(uiLayout *layout,
                                                    bContext *UNUSED(C),
                                                    PointerRNA *ptr)
{
  uiItemR(layout, ptr, "axis", UI_ITEM_R_EXPAND, nullptr, ICON_NONE);
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "pivot_axis", 0, IFACE_("Pivot"), ICON_NONE);
}

static void fn_node_align_rotation_to_vector_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  FunctionNodeAlignRotationToVector *node_storage = (FunctionNodeAlignRotationToVector *)
      MEM_callocN(sizeof(FunctionNodeAlignRotationToVector), __func__);
  node_storage->axis = GEO_NODE_ALIGN_ROTATION_TO_VECTOR_AXIS_X;
  node->storage = node_storage;
}

static float3 align_rotations_auto_pivot(const float3 vector,
                                         const float factor,
                                         const float3 local_main_axis,
                                         const float3 old_rotation_euler)
{
  if (is_zero_v3(vector)) {
    return float3(0);
  }

  float old_rotation[3][3];
  eul_to_mat3(old_rotation, old_rotation_euler);
  float3 old_axis;
  mul_v3_m3v3(old_axis, old_rotation, local_main_axis);

  const float3 new_axis = vector.normalized();
  float3 rotation_axis = float3::cross_high_precision(old_axis, new_axis);
  if (is_zero_v3(rotation_axis)) {
    /* The vectors are linearly dependent, so we fall back to another axis. */
    rotation_axis = float3::cross_high_precision(old_axis, float3(1, 0, 0));
    if (is_zero_v3(rotation_axis)) {
      /* This is now guaranteed to not be zero. */
      rotation_axis = float3::cross_high_precision(old_axis, float3(0, 1, 0));
    }
  }

  const float full_angle = angle_normalized_v3v3(old_axis, new_axis);
  const float angle = factor * full_angle;

  float rotation[3][3];
  axis_angle_to_mat3(rotation, rotation_axis, angle);

  float new_rotation_matrix[3][3];
  mul_m3_m3m3(new_rotation_matrix, rotation, old_rotation);

  float3 new_rotation;
  mat3_to_eul(new_rotation, new_rotation_matrix);

  return new_rotation;
}

static float3 align_rotations_fixed_pivot(const float3 vector,
                                          const float factor,
                                          const float3 local_main_axis,
                                          const float3 local_pivot_axis,
                                          const float3 old_rotation_euler)
{
  if (is_zero_v3(vector)) {
    return float3(0);
  }

  float old_rotation[3][3];
  eul_to_mat3(old_rotation, old_rotation_euler);
  float3 old_axis;
  mul_v3_m3v3(old_axis, old_rotation, local_main_axis);
  float3 pivot_axis;
  mul_v3_m3v3(pivot_axis, old_rotation, local_pivot_axis);

  float full_angle = angle_signed_on_axis_v3v3_v3(vector, old_axis, pivot_axis);
  if (full_angle > M_PI) {
    /* Make sure the point is rotated as little as possible. */
    full_angle -= 2.0f * M_PI;
  }
  const float angle = factor * full_angle;

  float rotation[3][3];
  axis_angle_to_mat3(rotation, pivot_axis, angle);

  float new_rotation_matrix[3][3];
  mul_m3_m3m3(new_rotation_matrix, rotation, old_rotation);

  float3 new_rotation;
  mat3_to_eul(new_rotation, new_rotation_matrix);

  return new_rotation;
}

static const blender::fn::MultiFunction &get_multi_function(bNode &node)
{
  const FunctionNodeAlignRotationToVector &storage = *(const FunctionNodeAlignRotationToVector *)
                                                          node.storage;

  float3 local_main_axis{0, 0, 0};
  local_main_axis[storage.axis] = 1;

  if (storage.pivot_axis == GEO_NODE_ALIGN_ROTATION_TO_VECTOR_PIVOT_AXIS_AUTO) {
    static blender::fn::CustomMF_SI_SI_SI_SO<float3, float, float3, float3> auto_pivot{
        "Align Rotation Auto Pivot",
        [local_main_axis](float3 rotation, float factor, float3 vector) {
          return align_rotations_auto_pivot(vector, factor, local_main_axis, rotation);
        }};
    return auto_pivot;
  }
  float3 local_pivot_axis{0, 0, 0};
  local_pivot_axis[storage.pivot_axis - 1] = 1;

  if (local_main_axis == local_pivot_axis) {
    return blender::fn::dummy_multi_function;
  }

  static blender::fn::CustomMF_SI_SI_SI_SO<float3, float, float3, float3> fixed_pivot{
      "Align Rotation Fixed Pivot",
      [local_main_axis, local_pivot_axis](float3 rotation, float factor, float3 vector) {
        return align_rotations_fixed_pivot(
            vector, factor, local_main_axis, local_pivot_axis, rotation);
      }};
  return fixed_pivot;
}

static void fn_node_align_rotation_to_vector_expand_in_mf_network(
    blender::nodes::NodeMFNetworkBuilder &builder)
{
  const blender::fn::MultiFunction &fn = get_multi_function(builder.bnode());
  builder.set_matching_fn(fn);
}

void register_node_type_fn_align_rotation_to_vector()
{
  static bNodeType ntype;

  fn_node_type_base(&ntype,
                    FN_NODE_ALIGN_ROTATION_TO_VECTOR,
                    "Align Rotation to Vector",
                    NODE_CLASS_OP_VECTOR,
                    0);
  node_type_socket_templates(
      &ntype, fn_node_align_rotation_to_vector_in, fn_node_align_rotation_to_vector_out);
  node_type_init(&ntype, fn_node_align_rotation_to_vector_init);
  node_type_storage(&ntype,
                    "FunctionNodeAlignRotationToVector",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  ntype.expand_in_mf_network = fn_node_align_rotation_to_vector_expand_in_mf_network;
  ntype.draw_buttons = fn_node_align_rotation_to_vector_layout;
  nodeRegisterType(&ntype);
}
