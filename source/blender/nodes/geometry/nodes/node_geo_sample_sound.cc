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

#include "UI_interface.h"
#include "UI_resources.h"

#include "DNA_sound_types.h"

#include "AUD_Sound.h"

#include "BKE_sound.h"

#include "DEG_depsgraph_query.h"

#include "node_geometry_util.hh"

namespace blender::nodes {

static void geo_node_sample_sound_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>(N_("Frame")).supports_field();
  b.add_input<decl::Float>(N_("Min Frequency")).supports_field().default_value(0.0f);
  b.add_input<decl::Float>(N_("Max Frequency")).supports_field().default_value(20000.0f);
  b.add_output<decl::Float>(N_("Volume")).dependent_field();
}

static void geo_node_sample_sound_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "sound", 0, "", ICON_NONE);
}

static void geo_node_sample_sound_exec(GeoNodeExecParams params)
{
  bSound *sound_id = (bSound *)params.node().id;
  Scene *scene = DEG_get_input_scene(params.depsgraph());
  const float fps = FPS;
  if (sound_id != nullptr) {
    AUD_Sound *sound = (AUD_Sound *)sound_id->handle;
    if (sound != NULL) {
      if (sound_id->samples == NULL) {
        AUD_Specs specs;
        sound_id->samples = AUD_Sound_data(sound, &sound_id->tot_samples, &specs);
      }
      AUD_Specs specs = AUD_Sound_getSpecs(sound);
      const float *samples = sound_id->samples;
      const int tot_samples = sound_id->tot_samples;
      const float sample_rate = specs.rate;

      const float frame = params.get_input<float>("Frame");
      const float time = (frame - 1) / fps;
      const int end_sample = std::max(
          0, std::min<int>(time * sample_rate * specs.channels, tot_samples));
      const int sample_duration = 1000;
      const int start_sample = std::max(0,
                                        std::min<int>(end_sample - sample_duration, tot_samples));
      float sum = 0.0f;
      for (int i = start_sample; i < end_sample; i++) {
        sum += fabsf(samples[i]);
      }
      const float average = sum / (end_sample - start_sample);
      params.set_output("Volume", average);
      return;
    }
  }
  params.set_output("Volume", 0.0f);
}

}  // namespace blender::nodes

void register_node_type_geo_sample_sound()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_SAMPLE_SOUND, "Sample Sound", NODE_CLASS_TEXTURE, 0);
  node_type_size(&ntype, 200, 40, 1000);
  ntype.declare = blender::nodes::geo_node_sample_sound_declare;
  ntype.geometry_node_execute = blender::nodes::geo_node_sample_sound_exec;
  ntype.draw_buttons = blender::nodes::geo_node_sample_sound_layout;
  nodeRegisterType(&ntype);
}
