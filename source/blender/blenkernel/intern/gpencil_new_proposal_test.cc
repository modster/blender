/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 */
#include "BKE_curves.hh"
#include "BLI_math_vec_types.hh"

#include "gpencil_new_proposal.hh"
#include "testing/testing.h"

template<typename T> class GPVector {
 private:
  /**
   * Address of the pointer to the begining of the vector.
   */
  T **start_;

  /**
   * Address of the size of the vector.
   */
  int *size_;

 public:
  GPVector(T **start, int *size) {
    this->start_ = start;
    this->size_ = size;
  }

  int size() {
    return size_;
  }

  T *data()
  {
    return *start_;
  }

  void append(T &elem) {

  }
 
 private:
  void realloc(const int64_t size)
  {
    /* Overwrite the size. */
    *this->size_ = size;
    /* Reallocate the array and overwrite the pointer to the beginning of the array. */
    *this->start_ = static_cast<T *>(MEM_reallocN(*this->start_, size * sizeof(T)));
  }

}

namespace blender::bke
{

  class GPLayerGroup : ::GPLayerGroup {
   public:
    GPLayerGroup()
    {
    }

    ~GPLayerGroup()
    {
      /* Recursivly free the children of this layer group first. */
      for (int i = 0; i < this->children_size; i++) {
        MEM_delete(&this->children[i]);
      }
      /* Then free its data. */
      MEM_SAFE_FREE(this->children);
      MEM_SAFE_FREE(this->layer_indices);
    }
  };

  class GPData : ::GPData {
   public:
    GPData()
    {
    }

    GPData(int layers_size)
    {
      BLI_assert(layers_size > 0);

      this->frames_size = 0;
      this->layers_size = layers_size;

      this->frames = nullptr;
      CustomData_reset(&this->frame_data);
      this->active_frame_index = -1;

      this->layers = (::GPLayer *)MEM_calloc_arrayN(this->layers_size, sizeof(::GPLayer), __func__);
      this->active_layer_index = 0;

      this->default_group = MEM_new<GPLayerGroup>(__func__);

      this->runtime = MEM_new<GPDataRuntime>(__func__);
    }

    ~GPData()
    {
      /* Free frames and frame custom data. */
      MEM_SAFE_FREE(this->frames);
      CustomData_free(&this->frame_data, this->frames_size);

      /* Free layer and layer groups. */
      MEM_SAFE_FREE(this->layers);
      MEM_delete(reinterpret_cast<GPLayerGroup *>(this->default_group));
      this->default_group = nullptr;

      MEM_delete(this->runtime) this->runtime = nullptr;
    }
  };

}  // namespace blender::bke

namespace blender::bke::gpencil::tests {

TEST(gpencil_proposal, Foo)
{
  GPData my_data;
  GPLayer my_layer("FooLayer");

  my_data.add_layer(my_layer);
  GPFrame my_frame = my_data.new_frame_on_layer(my_layer);
  my_frame.set_start_and_end(5, 10);

  GPStroke my_stroke(100);
  fill_stroke_with_points(my_stroke);
  my_frame.insert_stroke(my_stroke);
}

}  // namespace blender::bke::gpencil::tests