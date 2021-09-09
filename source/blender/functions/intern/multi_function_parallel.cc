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

#include "FN_multi_function_parallel.hh"

#include "BLI_task.hh"

#include <mutex>

namespace blender::fn {

ParallelMultiFunction::ParallelMultiFunction(const MultiFunction &fn, const int64_t grain_size)
    : fn_(fn), grain_size_(grain_size)
{
  this->set_signature(&fn.signature());

  threading_supported_ = true;
  for (const int param_index : fn.param_indices()) {
    const MFParamType param_type = fn.param_type(param_index);
    if (param_type.data_type().category() == MFDataType::Vector) {
      threading_supported_ = false;
      break;
    }
  }
}

void ParallelMultiFunction::call(IndexMask mask, MFParams params, MFContext context) const
{
  if (mask.size() <= grain_size_ || !threading_supported_) {
    fn_.call(mask, params, context);
    return;
  }

  threading::parallel_for(mask.index_range(), grain_size_, [&](const IndexRange range) {
    const int size = range.size();
    IndexMask original_sub_mask{mask.indices().slice(range)};
    const int64_t offset = original_sub_mask.indices().first();
    const int64_t slice_size = original_sub_mask.indices().last() - offset + 1;
    const IndexRange slice_range{offset, slice_size};
    IndexMask sub_mask;
    Vector<int64_t> sub_mask_indices;
    if (original_sub_mask.is_range()) {
      sub_mask = IndexMask(size);
    }
    else {
      sub_mask_indices.resize(size);
      for (const int i : IndexRange(size)) {
        sub_mask_indices[i] = original_sub_mask[i] - offset;
      }
      sub_mask = sub_mask_indices.as_span();
    }

    MFParamsBuilder sub_params{fn_, sub_mask.min_array_size()};
    ResourceScope scope;
    // static std::mutex mutex;
    // {
    //   std::lock_guard lock{mutex};
    //   std::cout << range << " " << sub_mask.min_array_size() << "\n";
    // }

    for (const int param_index : fn_.param_indices()) {
      const MFParamType param_type = fn_.param_type(param_index);
      switch (param_type.category()) {
        case MFParamType::SingleInput: {
          const GVArray &varray = params.readonly_single_input(param_index);
          const GVArray &sliced_varray = scope.construct<GVArray_Slice>(
              "sliced varray", varray, slice_range);
          sub_params.add_readonly_single_input(sliced_varray);
          break;
        }
        case MFParamType::SingleMutable: {
          const GMutableSpan span = params.single_mutable(param_index);
          const GMutableSpan sliced_span = span.slice(slice_range.start(), slice_range.size());
          sub_params.add_single_mutable(sliced_span);
          break;
        }
        case MFParamType::SingleOutput: {
          const GMutableSpan span = params.uninitialized_single_output(param_index);
          const GMutableSpan sliced_span = span.slice(slice_range.start(), slice_range.size());
          sub_params.add_uninitialized_single_output(sliced_span);
          break;
        }
        case MFParamType::VectorInput:
        case MFParamType::VectorMutable:
        case MFParamType::VectorOutput: {
          BLI_assert_unreachable();
          break;
        }
      }
    }

    fn_.call(sub_mask, sub_params, context);
  });
}

}  // namespace blender::fn
