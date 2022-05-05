/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 */

#include <algorithm>

#include "BKE_curves.hh"
#include "BLI_math_vec_types.hh"

#include "gpencil_new_proposal.hh"
#include "testing/testing.h"

namespace blender::bke {

class GPLayerGroup : ::GPLayerGroup {
 public:
  GPLayerGroup()
  {
    /* TODO */
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

class GPDataRuntime {
 public:
  /* mutable void *sbuffer */;
};

class GPLayer : public ::GPLayer {
 public:
  GPLayer() : GPLayer("GP_Layer")
  {
  }

  GPLayer(const StringRefNull name)
  {
    strcpy(this->name, name.c_str());
  }

  ~GPLayer() = default;

  bool operator==(const GPLayer &other) const
  {
    return STREQ(this->name, other.name);
  }
};

class GPFrame : public ::GPFrame {
 public:
  GPFrame()
  {
    this->layer_index = this->start = this->end = -1;
  }

  GPFrame(int layer_index)
  {
    this->layer_index = layer_index;
    this->start = this->end = -1;
  }

  GPFrame(int layer_index, int start_frame)
  {
    this->layer_index = layer_index;
    this->start = start_frame;
    this->end = -1;
  }

  ~GPFrame() = default;

  bool operator<(const GPFrame &other) const
  {
    if (this->start == other.start) {
      return this->layer_index < other.layer_index;
    }
    return this->start < other.start;
  }

  bool operator==(const GPFrame &other) const
  {
    return this->layer_index == other.layer_index && this->start == other.start;
  }
};

class GPData : public ::GPData {
 public:
  GPData() : GPData(0, 0)
  {
  }

  GPData(const int layers_size, const int frame_size)
  {
    BLI_assert(layers_size >= 0);
    BLI_assert(frame_size >= 0);

    this->frames_size = frame_size;
    this->layers_size = layers_size;

    if (this->frames_size > 0) {
      this->frames_array = reinterpret_cast<::GPFrame *>(
          MEM_calloc_arrayN(this->frames_size, sizeof(::GPFrame), __func__));
    }
    else {
      this->frames_array = nullptr;
    }
    CustomData_reset(&this->frame_data);

    if (this->layers_size > 0) {
      this->layers_array = reinterpret_cast<::GPLayer *>(
          MEM_calloc_arrayN(this->layers_size, sizeof(::GPLayer), __func__));
      this->active_layer_index = 0;
    }
    else {
      this->layers_array = nullptr;
      this->active_layer_index = -1;
    }

    this->default_group = MEM_new<::GPLayerGroup>(__func__);

    this->runtime = MEM_new<GPDataRuntime>(__func__);
  }

  ~GPData()
  {
    /* Free frames and frame custom data. */
    MEM_SAFE_FREE(this->frames_array);
    CustomData_free(&this->frame_data, this->frames_size);

    /* Free layer and layer groups. */
    MEM_SAFE_FREE(this->layers_array);
    MEM_delete(reinterpret_cast<GPLayerGroup *>(this->default_group));
    this->default_group = nullptr;

    MEM_delete(this->runtime);
    this->runtime = nullptr;
  }

  Span<GPFrame> frames() const
  {
    return {(const GPFrame *)this->frames_array, this->frames_size};
  }

  MutableSpan<GPFrame> frames_for_write()
  {
    return {(GPFrame *)this->frames_array, this->frames_size};
  }

  Span<GPLayer> layers() const
  {
    return {(const GPLayer *)this->layers_array, this->layers_size};
  }

  MutableSpan<GPLayer> layers_for_write()
  {
    return {(GPLayer *)this->layers_array, this->layers_size};
  }

  const bool add_layer(GPLayer &new_layer)
  {
    // Ensure that the layer array has enough space.
    if (!ensure_layer_array_has_size_at_least(this->layers_size + 1)) {
      return false;
    }

    // Move new_layer to the end in the array.
    this->layers_for_write().last() = new_layer;
    return true;
  }

  /**
   * Find a GPFrame in the frame_array given by the layer index and the frame number in logarithmic
   * time. If the frame is found, returns a pointer/iterator to it, otherwise nullptr.
   *
   * Note: This assumes that the array is sorted.
   */
  const GPFrame *frame_at(const int layer_idx, const int start_frame_number)
  {
    auto it = std::lower_bound(
        this->frames().begin(), this->frames().end(), GPFrame(layer_idx, start_frame_number));
    if (it == this->frames().end() || it->start != start_frame_number) {
      return nullptr;
    }
    return it;
  }

  void create_new_frame_on_layer(const int layer_index, const int start_frame_number)
  {
    BLI_assert(layer_index >= 0 && layer_index < this->layers_size);
    /* Allocate new space for the frame. */
    ensure_frame_array_has_size_at_least(this->frames_size + 1);

    /* Create a new frame and append it at the end. */
    GPFrame new_frame(layer_index, start_frame_number);
    this->frames_for_write().last() = new_frame;

    /* Sort the frame array. */
    update_frames_array();
  }

  void create_new_frame_on_layer(GPLayer &layer, const int start_frame_number)
  {
    int index = this->layers().first_index_try(layer);
    BLI_assert(index != -1);
    create_new_frame_on_layer(index, start_frame_number);
  }

  const GPFrame &get_new_frame_on_layer(const int layer_index, const int start_frame_number)
  {
    create_new_frame_on_layer(layer_index, start_frame_number);
    /* Find the frame in the array and return it. */
    const GPFrame *gpf = this->frame_at(layer_index, start_frame_number);
    BLI_assert(gpf != nullptr);
    return *gpf;
  }

  const GPFrame &get_new_frame_on_layer(GPLayer &layer, const int start_frame_number)
  {
    int index = this->layers().first_index_try(layer);
    BLI_assert(index != -1);
    return get_new_frame_on_layer(index, start_frame_number);
  }

 private:
  const bool ensure_layer_array_has_size_at_least(int64_t size)
  {
    if (this->layers_size > size) {
      return true;
    }

    int old_size = this->layers_size;
    this->layers_size = size;

    ::GPLayer *new_array = reinterpret_cast<::GPLayer *>(
        MEM_calloc_arrayN(this->layers_size, sizeof(::GPLayer), __func__));
    if (new_array == nullptr) {
      return false;
    }

    memcpy(new_array, this->layers_array, old_size * sizeof(::GPLayer));
    MEM_SAFE_FREE(this->layers_array);
    this->layers_array = new_array;

    return true;
  }

  const bool ensure_frame_array_has_size_at_least(int64_t size)
  {
    if (this->frames_size > size) {
      return true;
    }

    int old_size = this->frames_size;
    this->frames_size = size;

    ::GPFrame *new_array = reinterpret_cast<::GPFrame *>(
        MEM_calloc_arrayN(this->frames_size, sizeof(::GPFrame), __func__));
    if (new_array == nullptr) {
      return false;
    }

    memcpy(new_array, this->frames_array, old_size * sizeof(::GPFrame));
    MEM_SAFE_FREE(this->frames_array);
    this->frames_array = new_array;

    return true;
  }

  void update_frames_array()
  {
    /* Make sure frames are ordered chronologically and by layer order. */
    std::sort(this->frames_for_write().begin(), this->frames_for_write().end());
  }
};

}  // namespace blender::bke

namespace blender::bke::gpencil::tests {

TEST(gpencil_proposal, EmptyGPData)
{
  GPData my_data;
  EXPECT_EQ(my_data.layers_size, 0);
  EXPECT_EQ(my_data.frames_size, 0);
}

TEST(gpencil_proposal, OneLayer)
{
  GPData my_data(1, 0);
  EXPECT_EQ(my_data.layers_size, 1);
  EXPECT_EQ(my_data.frames_size, 0);
}

TEST(gpencil_proposal, LayerName)
{
  GPLayer my_layer1;
  EXPECT_STREQ(my_layer1.name, "GP_Layer");

  GPLayer my_layer2("FooLayer");
  EXPECT_STREQ(my_layer2.name, "FooLayer");
}

TEST(gpencil_proposal, AddOneLayer)
{
  GPData my_data;
  GPLayer my_layer("FooLayer");

  my_data.add_layer(my_layer);
  EXPECT_EQ(my_data.layers_size, 1);
  EXPECT_STREQ(my_data.layers().last().name, my_layer.name);
}

TEST(gpencil_proposal, AddLayers)
{
  GPData my_data;
  GPLayer layers[3] = {GPLayer("TestLayer1"), GPLayer("TestLayer2"), GPLayer("TestLayer3")};

  for (int i : IndexRange(3)) {
    my_data.add_layer(layers[i]);
  }
  EXPECT_EQ(my_data.layers_size, 3);

  for (int i : IndexRange(3)) {
    EXPECT_STREQ(my_data.layers()[i].name, layers[i].name);
  }
}

TEST(gpencil_proposal, ChangeLayerName)
{
  GPData my_data;
  GPLayer my_layer("FooLayer");

  my_data.add_layer(my_layer);
  EXPECT_EQ(my_data.layers_size, 1);
  EXPECT_STREQ(my_data.layers().last().name, my_layer.name);

  strcpy(my_data.layers_for_write().last().name, "BarLayer");

  EXPECT_EQ(my_data.layers_size, 1);
  EXPECT_STREQ(my_data.layers().last().name, "BarLayer");
}

TEST(gpencil_proposal, AddFrameToLayer)
{
  GPData my_data;
  GPLayer my_layer1("TestLayer1");
  GPLayer my_layer2("TestLayer2");

  my_data.add_layer(my_layer1);
  my_data.add_layer(my_layer2);
  GPFrame my_frame = my_data.get_new_frame_on_layer(my_layer2, 0);
  EXPECT_EQ(my_frame.layer_index, 1);
}

TEST(gpencil_proposal, CheckFramesSorted1)
{
  GPData my_data;
  GPLayer my_layer1("TestLayer1");

  const int frame_numbers1[5] = {10, 5, 6, 1, 3};
  const int frame_numbers_sorted1[5] = {1, 3, 5, 6, 10};

  my_data.add_layer(my_layer1);
  for (int i : IndexRange(5)) {
    GPFrame my_frame = my_data.get_new_frame_on_layer(my_layer1, frame_numbers1[i]);
    int idx = my_data.frames().first_index(my_frame);
    EXPECT_EQ(my_data.frames()[idx].start, frame_numbers1[i]);
  }

  for (const int i : my_data.frames().index_range()) {
    EXPECT_EQ(my_data.frames()[i].start, frame_numbers_sorted1[i]);
  }
}

TEST(gpencil_proposal, CheckFramesSorted2)
{
  GPData my_data;
  GPLayer my_layer1("TestLayer1");
  GPLayer my_layer2("TestLayer2");
  const int frame_numbers_layer1[5] = {10, 5, 6, 1, 3};
  const int frame_numbers_layer2[5] = {8, 5, 7, 1, 4};
  const int frame_numbers_sorted2[10][2] = {
      {0, 1}, {1, 1}, {0, 3}, {1, 4}, {0, 5}, {1, 5}, {0, 6}, {1, 7}, {1, 8}, {0, 10}};

  my_data.add_layer(my_layer1);
  my_data.add_layer(my_layer2);
  for (int i : IndexRange(5)) {
    my_data.create_new_frame_on_layer(my_layer1, frame_numbers_layer1[i]);
    my_data.create_new_frame_on_layer(my_layer2, frame_numbers_layer2[i]);
  }

  for (const int i : my_data.frames().index_range()) {
    EXPECT_EQ(my_data.frames()[i].layer_index, frame_numbers_sorted2[i][0]);
    EXPECT_EQ(my_data.frames()[i].start, frame_numbers_sorted2[i][1]);
  }
}

}  // namespace blender::bke::gpencil::tests