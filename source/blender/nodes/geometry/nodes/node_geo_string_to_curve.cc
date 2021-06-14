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

#include "DNA_curve_types.h"
#include "DNA_vfont_types.h"

#include "BKE_curve.h"
#include "BKE_font.h"
#include "BKE_spline.hh"

#include "BLI_string_utf8.h"

#include "UI_interface.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_string_to_curve_in[] = {
    {SOCK_FLOAT, N_("Size"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, PROP_DISTANCE},
    {SOCK_FLOAT, N_("Spacing"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, PROP_DISTANCE},
    {SOCK_FLOAT, N_("Line Distance"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, PROP_DISTANCE},
    {SOCK_STRING, N_("String")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_string_to_curve_out[] = {
    {SOCK_GEOMETRY, N_("Curve")},
    {-1, ""},
};

static void geo_node_string_to_curve_layout(uiLayout *layout, struct bContext *C, PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiTemplateID(layout,
               C,
               ptr,
               "font",
               nullptr,
               "FONT_OT_open",
               "FONT_OT_unlink",
               UI_TEMPLATE_ID_FILTER_ALL,
               false,
               nullptr);
}

namespace blender::nodes {

static void geo_node_string_to_curve_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->id = (ID *)BKE_vfont_builtin_get();
}

Curve create_text_curve(StringRef text, float size, float spacing, float linedist, VFont *vfont)
{
  Curve cu = {0};
  cu.type = OB_FONT;
  /* Set defaults */
  cu.spacemode = CU_ALIGN_X_LEFT;
  cu.resolu = 12;
  cu.smallcaps_scale = 0.75f;
  cu.wordspace = 1.0f;
  /* Set values from inputs */
  cu.fsize = size;
  cu.linedist = linedist;
  cu.spacing = spacing;
  cu.vfont = vfont;

  size_t len_bytes;
  size_t len_chars = BLI_strlen_utf8_ex(text.data(), &len_bytes);
  cu.len_char32 = len_chars;
  cu.len = len_bytes;
  cu.pos = len_chars;
  cu.str = (char *)MEM_mallocN(len_bytes + sizeof(char32_t), "str");
  cu.strinfo = (CharInfo *)MEM_callocN((len_chars + 4) * sizeof(CharInfo), "strinfo");

  BLI_strncpy(cu.str, text.data(), len_bytes + 1);
  BKE_vfont_to_curve_ex(nullptr, &cu, FO_EDIT, &cu.nurb, nullptr, nullptr, nullptr, nullptr);
  return cu;
}

void free_text_curve(Curve &cu)
{
  BKE_nurbList_free(&cu.nurb);
  MEM_SAFE_FREE(cu.str);
  MEM_SAFE_FREE(cu.strinfo);
  MEM_SAFE_FREE(cu.tb);
}

std::unique_ptr<CurveEval> curve_eval_text(
    StringRef text, float size, float spacing, float linedist, VFont *vfont)
{
  if (text.is_empty()) {
    return nullptr;
  }
  Curve cu = create_text_curve(text, size, spacing, linedist, vfont);
  std::unique_ptr<CurveEval> curve_eval = curve_eval_from_dna_curve(cu);
  free_text_curve(cu);
  return curve_eval;
}

void string_replace(std::string &haystack, const std::string &needle, const std::string &other)
{
  if (&haystack == nullptr || needle.empty())
    return;
  size_t i = 0, index;
  while ((index = haystack.find(needle, i)) != std::string::npos) {
    haystack.replace(index, needle.length(), other);
    i = index + other.length();
  }
}

void parse_text(std::string &text)
{
  /* Replace typed in "\n" with newline */
  string_replace(text, "\\n", "\n");
}

static void geo_node_string_to_curve_exec(GeoNodeExecParams params)
{
  std::string text = params.extract_input<std::string>("String");
  const float size = params.extract_input<float>("Size");
  const float spacing = params.extract_input<float>("Spacing");
  const float linedist = params.extract_input<float>("Line Distance");

  parse_text(text);
  VFont *vfont = (VFont *)params.node().id;
  std::unique_ptr<CurveEval> curve = curve_eval_text(text, size, spacing, linedist, vfont);

  params.set_output("Curve", GeometrySet::create_with_curve(curve.release()));
}

}  // namespace blender::nodes

void register_node_type_geo_string_to_curve()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_STRING_TO_CURVE, "String to Curve", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_string_to_curve_in, geo_node_string_to_curve_out);
  node_type_init(&ntype, blender::nodes::geo_node_string_to_curve_init);
  ntype.geometry_node_execute = blender::nodes::geo_node_string_to_curve_exec;
  ntype.draw_buttons = geo_node_string_to_curve_layout;
  nodeRegisterType(&ntype);
}
