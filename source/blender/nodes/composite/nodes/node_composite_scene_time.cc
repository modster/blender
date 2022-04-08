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
 * \ingroup cmpnodes
 */

#include "VPC_compositor_execute.hh"

#include "node_composite_util.hh"

namespace blender::nodes {

static void cmp_node_scene_time_declare(NodeDeclarationBuilder &b)
{
  b.add_output<decl::Float>(N_("Seconds"));
  b.add_output<decl::Float>(N_("Frame"));
}

using namespace blender::viewport_compositor;

class SceneTimeOperation : public NodeOperation {
 public:
  using NodeOperation::NodeOperation;

  void execute() override
  {
    execute_seconds();
    execute_frame();
  }

  void execute_seconds()
  {
    Result &result = get_result("Seconds");
    result.allocate_single_value();

    const int frame_number = static_cast<float>(context().get_scene()->r.cfra);
    const float frame_rate = static_cast<float>(context().get_scene()->r.frs_sec) /
                             static_cast<float>(context().get_scene()->r.frs_sec_base);

    result.set_float_value(frame_number / frame_rate);
  }

  void execute_frame()
  {
    Result &result = get_result("Frame");
    result.allocate_single_value();

    const int frame_number = static_cast<float>(context().get_scene()->r.cfra);

    result.set_float_value(frame_number);
  }
};

static NodeOperation *get_compositor_operation(Context &context, DNode node)
{
  return new SceneTimeOperation(context, node);
}

}  // namespace blender::nodes

void register_node_type_cmp_scene_time()
{
  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_SCENE_TIME, "Scene Time", NODE_CLASS_INPUT);
  ntype.declare = blender::nodes::cmp_node_scene_time_declare;
  ntype.get_compositor_operation = blender::nodes::get_compositor_operation;

  nodeRegisterType(&ntype);
}
