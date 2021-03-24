/*
 * Copyright 2011-2018 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "graph/node.h"
#include "render/attribute.h"
#include "render/procedural.h"
#include "util/util_set.h"
#include "util/util_task.h"
#include "util/util_transform.h"
#include "util/util_vector.h"

#ifdef WITH_ALEMBIC

#  include <Alembic/AbcCoreFactory/All.h>
#  include <Alembic/AbcGeom/All.h>

CCL_NAMESPACE_BEGIN

class AlembicProcedural;
class Geometry;
class Object;
class Progress;
class Shader;

using MatrixSampleMap = std::map<Alembic::Abc::chrono_t, Alembic::Abc::M44d>;

struct MatrixSamplesData {
  MatrixSampleMap *samples = nullptr;
  Alembic::AbcCoreAbstract::TimeSamplingPtr time_sampling;
};

/* Helpers to detect if some type is a `ccl::array`. */
template<typename> struct is_array : public std::false_type {
};

template<typename T> struct is_array<array<T>> : public std::true_type {
};

/* Holds the data for a cache lookup at a given time, as well as information to
 * help disambiguate successes or failures to get data from the cache. */
template<typename T> class CacheLookupResult {
  enum class State {
    NEW_DATA,
    ALREADY_LOADED,
    NO_DATA_FOR_TIME,
  };

  T *data;
  State state;

 protected:
  /* Prevent default construction outside of the class: for a valid result, we
   * should use the static functions below. */
  CacheLookupResult() = default;

 public:
  static CacheLookupResult new_data(T *data_)
  {
    CacheLookupResult result;
    result.data = data_;
    result.state = State::NEW_DATA;
    return result;
  }

  static CacheLookupResult no_data_found_for_time()
  {
    CacheLookupResult result;
    result.data = nullptr;
    result.state = State::NO_DATA_FOR_TIME;
    return result;
  }

  static CacheLookupResult already_loaded()
  {
    CacheLookupResult result;
    result.data = nullptr;
    result.state = State::ALREADY_LOADED;
    return result;
  }

  /* This should only be call if new data is available. */
  const T &get_data() const
  {
    assert(state == State::NEW_DATA);
    assert(data != nullptr);
    return *data;
  }

  T *get_data_or_null() const
  {
    // data_ should already be null if there is no new data so no need to check
    return data;
  }

  bool has_new_data() const
  {
    return state == State::NEW_DATA;
  }

  bool has_already_loaded() const
  {
    return state == State::ALREADY_LOADED;
  }

  bool has_no_data_for_time() const
  {
    return state == State::NO_DATA_FOR_TIME;
  }
};

/* Store the data set for an animation at every time points, or at the beginning of the animation
 * for constant data.
 *
 * The data is supposed to be stored in chronological order, and is looked up using the current
 * animation time in seconds using the TimeSampling from the Alembic property. */
template<typename T> class DataStore {
  /* Holds information to map a cache entry for a given time to an index into the data array. */
  struct TimeIndexPair {
    /* Frame time for this entry. */
    double time = 0;
    /* Frame time for the data pointed to by `index`. */
    double source_time = 0;
    /* Index into the data array. */
    size_t index = 0;
  };

  /* This is the actual data that is stored. We deduplicate data across frames to avoid storing
   * values if they have not changed yet (e.g. the triangles for a building before fracturing, or a
   * fluid simulation before a break or splash) */
  vector<T> data{};

  /* This is used to map they entry for a given time to an index into the data array, multiple
   * frames can point to the same index. */
  vector<TimeIndexPair> index_data_map{};

  Alembic::AbcCoreAbstract::TimeSampling time_sampling{};

  double last_loaded_time = std::numeric_limits<double>::max();
  int frame_offset = 0;

 public:
  void set_time_sampling(Alembic::AbcCoreAbstract::TimeSampling time_sampling_, int frame_offset_)
  {
    time_sampling = time_sampling_;
    frame_offset = frame_offset_;
  }

  Alembic::AbcCoreAbstract::TimeSampling get_time_sampling() const
  {
    return time_sampling;
  }

  /* Get the data for the specified time.
   * Return nullptr if there is no data or if the data for this time was already loaded. */
  CacheLookupResult<T> data_for_time(double time)
  {
    if (size() == 0) {
      return CacheLookupResult<T>::no_data_found_for_time();
    }

    const TimeIndexPair &index = get_index_for_time(time);

    if (index.index == -1ul) {
      return CacheLookupResult<T>::no_data_found_for_time();
    }

    if (last_loaded_time == index.time || last_loaded_time == index.source_time) {
      return CacheLookupResult<T>::already_loaded();
    }

    last_loaded_time = index.source_time;

    assert(index.index < data.size());

    return CacheLookupResult<T>::new_data(&data[index.index]);
  }

  /* get the data for the specified time, but do not check if the data was already loaded for this
   * time return nullptr if there is no data */
  CacheLookupResult<T> data_for_time_no_check(double time)
  {
    if (size() == 0) {
      return CacheLookupResult<T>::no_data_found_for_time();
    }

    const TimeIndexPair &index = get_index_for_time(time);

    if (index.index == -1ul) {
      return CacheLookupResult<T>::no_data_found_for_time();
    }

    assert(index.index < data.size());

    return CacheLookupResult<T>::new_data(&data[index.index]);
  }

  void add_data(T &data_, double time)
  {
    index_data_map.push_back({time, time, data.size()});

    if constexpr (is_array<T>::value) {
      data.emplace_back();
      data.back().steal_data(data_);
      return;
    }

    data.push_back(data_);
  }

  void reuse_data_for_last_time(double time)
  {
    const TimeIndexPair &data_index = index_data_map.back();
    index_data_map.push_back({time, data_index.source_time, data_index.index});
  }

  void add_no_data(double time)
  {
    index_data_map.push_back({time, time, -1ul});
  }

  bool is_constant() const
  {
    return data.size() <= 1;
  }

  size_t size() const
  {
    return data.size();
  }

  void clear()
  {
    invalidate_last_loaded_time();
    data.clear();
  }

  void invalidate_last_loaded_time()
  {
    last_loaded_time = std::numeric_limits<double>::max();
  }

  /* Copy the data for the specified time to the node's socket. If there is no
   * data for this time or it was already loaded, do nothing. */
  void copy_to_socket(double time, Node *node, const SocketType *socket)
  {
    CacheLookupResult<T> result = data_for_time(time);

    if (!result.has_new_data()) {
      return;
    }

    /* TODO(kevindietrich): arrays are emptied when passed to the sockets, so for now we copy the
     * arrays to avoid reloading the data */
    T value = result.get_data();
    node->set(*socket, value);
  }

  size_t memory_used() const
  {
    if constexpr (is_array<T>::value) {
      size_t mem_used = 0;

      for (const T &array : data) {
        mem_used += array.size() * sizeof(array[0]);
      }

      return mem_used;
    }

    return data.size() * sizeof(T);
  }

  void swap(DataStore<T> &other)
  {
    if (this == &other) {
      return;
    }

    index_data_map.swap(other.index_data_map);
    data.swap(other.data);
    std::swap(frame_offset, other.frame_offset);
  }

 private:
  const TimeIndexPair &get_index_for_time(double time) const
  {
    /* TimeSampling works by matching a frame time to an index, however it expects
     * frame time 0 == index 0, but since we may load data by chunks of frames,
     * index 0 may not be frame time 0, so we need to offset the size to pretend
     * we have the required amount of frame data. This offset should only be applied
     * if we have more than one frame worth of data (for now we it is guaranteed that
     * we either have dat for one single frame, or for all the frames of the animation,
     * this may change in the future). */
    size_t size_offset = 0;
    if (size() != 1) {
      size_offset = frame_offset;
    }

    std::pair<size_t, Alembic::Abc::chrono_t> index_pair;
    index_pair = time_sampling.getNearIndex(time, index_data_map.size() + size_offset);
    return index_data_map[index_pair.first - size_offset];
  }
};

/* Actual cache for the stored data.
 * This caches the topological, transformation, and attribute data for a Mesh node or a Hair node
 * inside of DataStores.
 */
struct CachedData {
  DataStore<Transform> transforms{};

  /* mesh data */
  DataStore<array<float3>> vertices;
  DataStore<array<int3>> triangles{};
  /* triangle "loops" are the polygons' vertices indices used for indexing face varying attributes
   * (like UVs) */
  DataStore<array<int3>> triangles_loops{};
  DataStore<array<int>> shader{};

  /* subd data */
  DataStore<array<int>> subd_start_corner;
  DataStore<array<int>> subd_num_corners;
  DataStore<array<bool>> subd_smooth;
  DataStore<array<int>> subd_ptex_offset;
  DataStore<array<int>> subd_face_corners;
  DataStore<int> num_ngons;
  DataStore<array<int>> subd_creases_edge;
  DataStore<array<float>> subd_creases_weight;

  /* hair data */
  DataStore<array<float3>> curve_keys;
  DataStore<array<float>> curve_radius;
  DataStore<array<int>> curve_first_key;
  DataStore<array<int>> curve_shader;

  /* ranges for delta compression, values should be in sync with the attribute */
  DataStore<float> min_delta;
  DataStore<float> max_delta;

  struct CachedAttribute {
    AttributeStandard std;
    AttributeElement element;
    TypeDesc type_desc;
    ustring name;
    DataStore<array<char>> data{};
  };

  vector<CachedAttribute> attributes{};

  int frame_start = -1;
  int frame_end = -1;
  int frame_offset = 0;

  void clear();

  CachedAttribute &add_attribute(const ustring &name,
                                 const Alembic::Abc::TimeSampling &time_sampling);

  bool is_constant() const;

  void invalidate_last_loaded_time(bool attributes_only = false);

  void set_time_sampling(Alembic::AbcCoreAbstract::TimeSampling time_sampling);

  size_t memory_used() const;

  void swap(CachedData &other);
};

/* Representation of an Alembic object for the AlembicProcedural.
 *
 * The AlembicObject holds the path to the Alembic IObject inside of the archive that is desired
 * for rendering, as well as the list of shaders that it is using.
 *
 * The names of the shaders should correspond to the names of the FaceSets inside of the Alembic
 * archive for per-triangle shader association. If there is no FaceSets, or the names do not
 * match, the first shader is used for rendering for all triangles.
 */
class AlembicObject : public Node {
 public:
  NODE_DECLARE

  /* Path to the IObject inside of the archive. */
  NODE_SOCKET_API(ustring, path)

  /* Shaders used for rendering. */
  NODE_SOCKET_API_ARRAY(array<Node *>, used_shaders)

  /* Maximum number of subdivisions for ISubD objects. */
  NODE_SOCKET_API(int, subd_max_level)

  /* Finest level of detail (in pixels) for the subdivision. */
  NODE_SOCKET_API(float, subd_dicing_rate)

  /* Scale the radius of points and curves. */
  NODE_SOCKET_API(float, radius_scale)

  AlembicObject();
  ~AlembicObject();

 private:
  friend class AlembicProcedural;

  void set_object(Object *object);
  Object *get_object();

  bool load_all_data(CachedData &cached_data,
                     AlembicProcedural *proc,
                     const int frame,
                     Alembic::AbcGeom::IPolyMeshSchema &schema,
                     Progress &progress,
                     bool for_prefetch);
  bool load_all_data(CachedData &cached_data,
                     AlembicProcedural *proc,
                     const int frame,
                     Alembic::AbcGeom::ISubDSchema &schema,
                     Progress &progress,
                     bool for_prefetch);
  bool load_all_data(CachedData &cached_data,
                     AlembicProcedural *proc,
                     const int frame,
                     const Alembic::AbcGeom::ICurvesSchema &schema,
                     Progress &progress,
                     float default_radius,
                     bool for_prefetch);

  bool has_data_loaded(int frame) const;

  /* Enumeration used to speed up the discrimination of an IObject as IObject::matches() methods
   * are too expensive and show up in profiles. */
  enum AbcSchemaType {
    INVALID,
    POLY_MESH,
    SUBD,
    CURVES,
  };

  bool need_shader_update = true;

  AlembicObject *instance_of = nullptr;

  Alembic::AbcCoreAbstract::TimeSamplingPtr xform_time_sampling;
  MatrixSampleMap xform_samples;
  Alembic::AbcGeom::IObject iobject;

  CachedData &get_cached_data()
  {
    return cached_data_;
  }

  bool is_constant() const
  {
    return cached_data_.is_constant();
  }

  void clear_all_caches()
  {
    cached_data_.clear();

    if (prefetch_cache) {
      prefetch_cache->clear();
      delete prefetch_cache;
      prefetch_cache = nullptr;
    }

    data_loaded = false;
  }

  Object *object = nullptr;

  bool data_loaded = false;

  /* Set on construction. */
  AbcSchemaType schema_type;

  CachedData cached_data_;
  /* cache used to prefetch the next N frames during rendering */
  CachedData *prefetch_cache;

  void update_shader_attributes(CachedData &cached_data,
                                const Alembic::AbcGeom::ICompoundProperty &arb_geom_params,
                                Progress &progress);

  void read_attribute(CachedData &cached_data,
                      const Alembic::AbcGeom::ICompoundProperty &arb_geom_params,
                      const ustring &attr_name,
                      Progress &progress);

  template<typename SchemaType>
  void read_face_sets(SchemaType &schema,
                      array<int> &polygon_to_shader,
                      Alembic::AbcGeom::ISampleSelector sample_sel);

  void setup_transform_cache(CachedData &cached_data, float scale);

  AttributeRequestSet get_requested_attributes();

  void swap_prefetch_cache();
};

/* Procedural to render objects from a single Alembic archive.
 *
 * Every object desired to be rendered should be passed as an AlembicObject through the objects
 * socket.
 *
 * This procedural will load the data set for the entire animation in memory on the first frame,
 * and directly set the data for the new frames on the created Nodes if needed. This allows for
 * faster updates between frames as it avoids reseeking the data on disk.
 */
class AlembicProcedural : public Procedural {
  Alembic::AbcGeom::IArchive archive;
  bool objects_loaded;
  Scene *scene_;

  DedicatedTaskPool prefetch_pool;

 public:
  NODE_DECLARE

  /* The file path to the Alembic archive */
  NODE_SOCKET_API(ustring, filepath)

  /* The current frame to render. */
  NODE_SOCKET_API(float, frame)

  /* The first frame to load data for. */
  NODE_SOCKET_API(float, start_frame)

  /* The last frame to load data for. */
  NODE_SOCKET_API(float, end_frame)

  /* Subtracted to the current frame. */
  NODE_SOCKET_API(float, frame_offset)

  /* The frame rate used for rendering in units of frames per second. */
  NODE_SOCKET_API(float, frame_rate)

  /* List of AlembicObjects to render. */
  NODE_SOCKET_API_ARRAY(array<Node *>, objects)

  /* Set the default radius to use for curves when the Alembic Curves Schemas do not have radius
   * information. */
  NODE_SOCKET_API(float, default_radius)

  /* Multiplier to account for differences in default units for measuring objects in various
   * software. */
  NODE_SOCKET_API(float, scale)

  /* Cache control. */

  enum CacheMethod {
    NO_CACHE,
    CACHE_FRAME_COUNT,
    CACHE_ALL_DATA,
  };

  NODE_SOCKET_API(int, cache_method)

  /* Maximum number of frames to hold in cache. */
  NODE_SOCKET_API(int, cache_frame_count)

  /* Whether to preload data in a secondary cache, only valid if cache method is CACHE_FRAME_COUNT.
   */
  NODE_SOCKET_API(bool, use_prefetching)

  /* Treat subdivision objects as regular polygon meshes. */
  NODE_SOCKET_API(bool, ignore_subdivision)

  AlembicProcedural();
  ~AlembicProcedural();

  /* Populates the Cycles scene with Nodes for every contained AlembicObject on the first
   * invocation, and updates the data on subsequent invocations if the frame changed. */
  void generate(Scene *scene, Progress &progress);

  /* Add an object to our list of objects, and tag the socket as modified. */
  void add_object(AlembicObject *object);

  /* Tag for an update only if something was modified. */
  void tag_update(Scene *scene);

  /* Returns a pointer to an existing or a newly created AlembicObject for the given path. */
  AlembicObject *get_or_create_object(const ustring &path);

 private:
  /* Load the data for all the objects whose data has not yet been loaded. */
  void load_objects(Progress &progress);

  /* Traverse the Alembic hierarchy to lookup the IObjects for the AlembicObjects that were
   * specified in our objects socket, and accumulate all of the transformations samples along the
   * way for each IObject. */
  void walk_hierarchy(Alembic::AbcGeom::IObject parent,
                      const Alembic::AbcGeom::ObjectHeader &ohead,
                      MatrixSamplesData matrix_samples_data,
                      const unordered_map<string, AlembicObject *> &object_map,
                      Progress &progress);

  /* Read the data for an IPolyMesh at the specified frame_time. Creates corresponding Geometry and
   * Object Nodes in the Cycles scene if none exist yet. */
  void read_mesh(AlembicObject *abc_object, Alembic::AbcGeom::Abc::chrono_t frame_time);

  /* Read the data for an ICurves at the specified frame_time. Creates corresponding Geometry and
   * Object Nodes in the Cycles scene if none exist yet. */
  void read_curves(AlembicObject *abc_object, Alembic::AbcGeom::Abc::chrono_t frame_time);

  /* Read the data for an ISubD at the specified frame_time. Creates corresponding Geometry and
   * Object Nodes in the Cycles scene if none exist yet. */
  void read_subd(AlembicObject *abc_object, Alembic::AbcGeom::Abc::chrono_t frame_time);

  void build_caches(Progress &progress);
};

CCL_NAMESPACE_END

#endif
