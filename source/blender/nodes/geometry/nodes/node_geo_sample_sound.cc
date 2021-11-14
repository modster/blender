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

#include <fftw3.h>

#include "UI_interface.h"
#include "UI_resources.h"

#include "BLI_double2.hh"

#include "DNA_sound_types.h"

#include "AUD_Sound.h"

#include "BKE_sound.h"

#include "DEG_depsgraph_query.h"

#include "node_geometry_util.hh"

namespace blender::nodes {

static void geo_node_sample_sound_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>(N_("Time")).supports_field();
  b.add_input<decl::Float>(N_("Min Frequency")).supports_field().default_value(0.0f).min(0.0f);
  b.add_input<decl::Float>(N_("Max Frequency")).supports_field().default_value(20000.0f).min(0.0f);
  b.add_input<decl::Int>(N_("Channel")).min(0);
  b.add_output<decl::Float>(N_("Volume")).dependent_field();
}

static void geo_node_sample_sound_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "sound", 0, "", ICON_NONE);
}

struct SoundData {
  Span<float> samples;
  float sample_rate;
  int channels;
};

class SampleSoundFunction : public fn::MultiFunction {
 private:
  SoundData sound_data_;

 public:
  SampleSoundFunction(SoundData sound_data) : sound_data_(sound_data)
  {
    static fn::MFSignature signature = create_signature();
    this->set_signature(&signature);
  }

  static fn::MFSignature create_signature()
  {
    fn::MFSignatureBuilder signature{"Sample Sound"};
    signature.single_input<float>("Time");
    signature.single_input<float>("Min Frequency");
    signature.single_input<float>("Max Frequency");
    signature.single_input<int>("Channel");
    signature.single_output<float>("Volume");
    return signature.build();
  }

  void call(IndexMask mask, fn::MFParams params, fn::MFContext UNUSED(context)) const override
  {
    const VArray<float> &times = params.readonly_single_input<float>(0, "Time");
    const VArray<float> &min_frequencies = params.readonly_single_input<float>(1,
                                                                               "Min Frequencies");
    const VArray<float> &max_frequencies = params.readonly_single_input<float>(2,
                                                                               "Max Frequencies");
    const VArray<int> &channels = params.readonly_single_input<int>(3, "Channel");
    MutableSpan<float> r_volumes = params.uninitialized_single_output<float>(4, "Volume");

    /* TODO: Remove this limitation. */
    BLI_assert(times.is_single());
    BLI_assert(channels.is_single());

    if (mask.is_empty()) {
      return;
    }

    const float time = times.get_internal_single();
    const int channel = channels.get_internal_single();

    const int desired_slice_size = 4000;
    int end_sample = time * sound_data_.sample_rate;
    int start_sample = end_sample - desired_slice_size;
    CLAMP(start_sample, 0, sound_data_.samples.size());
    CLAMP(end_sample, 0, sound_data_.samples.size());
    const int slice_size = end_sample - start_sample;

    if (slice_size == 0) {
      r_volumes.fill_indices(mask, 0.0f);
      return;
    }

    Array<double> raw_samples(slice_size);
    Array<double2> frequencies(slice_size / 2 + 1);

    fftw_plan plan = fftw_plan_dft_r2c_1d(
        slice_size, raw_samples.data(), (fftw_complex *)frequencies.data(), 0);

    for (const int i : IndexRange(slice_size)) {
      raw_samples[i] = sound_data_.samples[(start_sample + i) * sound_data_.channels + channel];
    }

    fftw_execute(plan);
    fftw_destroy_plan(plan);

    const float band_per_index = sound_data_.sample_rate / slice_size;
    auto frequency_to_index = [&](const float frequency) -> int {
      return frequency / band_per_index;
    };

    for (const int i : mask) {
      const float min_frequency = min_frequencies[i];
      const float max_frequency = max_frequencies[i];
      int min_index = frequency_to_index(min_frequency);
      int max_index = frequency_to_index(max_frequency);
      min_index = std::clamp<int>(min_index, 0, frequencies.size());
      max_index = std::clamp<int>(max_index, min_index, frequencies.size());

      float sum = 0.0f;
      for (int frequency_index = min_index; frequency_index < max_index; frequency_index++) {
        sum += fabsf(frequencies[frequency_index].x);
      }
      const float average = (max_index > min_index) ? sum / (max_index - min_index) : 0.0f;
      r_volumes[i] = average;
    }
  }
};

static void geo_node_sample_sound_exec(GeoNodeExecParams params)
{
  bSound *sound_id = (bSound *)params.node().id;
  if (sound_id != nullptr) {
    AUD_Sound *sound = (AUD_Sound *)sound_id->handle;
    if (sound != NULL) {
      if (sound_id->samples == NULL) {
        AUD_Specs specs;
        sound_id->samples = AUD_Sound_data(sound, &sound_id->tot_samples, &specs);
      }
      AUD_Specs specs = AUD_Sound_getSpecs(sound);
      SoundData sound_data;
      sound_data.channels = specs.channels;
      sound_data.sample_rate = specs.rate;
      sound_data.samples = Span<float>(sound_id->samples, sound_id->tot_samples);
      auto sample_fn = std::make_shared<SampleSoundFunction>(sound_data);
      auto sample_node = std::make_shared<FieldOperation>(
          sample_fn,
          Vector<GField>{params.extract_input<Field<float>>("Time"),
                         params.extract_input<Field<float>>("Min Frequency"),
                         params.extract_input<Field<float>>("Max Frequency"),
                         params.extract_input<Field<int>>("Channel")});
      Field<float> output_field{std::move(sample_node), 0};
      params.set_output("Volume", std::move(output_field));
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
