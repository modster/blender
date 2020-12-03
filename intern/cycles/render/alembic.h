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
#include "util/util_transform.h"
#include "util/util_vector.h"

#ifdef WITH_ALEMBIC

#  include <Alembic/AbcCoreFactory/All.h>
#  include <Alembic/AbcGeom/All.h>

CCL_NAMESPACE_BEGIN

class Geometry;
class Object;
class Progress;
class Shader;

using MatrixSampleMap = std::map<Alembic::Abc::chrono_t, Alembic::Abc::M44d>;

template<typename T> class DataStore {
  struct DataTimePair {
    double time = 0;
    T data{};
  };

  vector<DataTimePair> data{};

 public:
  Alembic::AbcCoreAbstract::TimeSampling time_sampling{};

  void set_time_sampling(Alembic::AbcCoreAbstract::TimeSampling time_sampling_)
  {
    time_sampling = time_sampling_;
  }

  T *data_for_time(double time)
  {
    if (size() == 0) {
      return nullptr;
    }

    auto index_pair = time_sampling.getNearIndex(time, data.size());
    auto ptr = &data[index_pair.first];

    return &ptr->data;
  }

  void add_data(T data_, double time)
  {
    data.push_back({time, data_});
  }

  void add_data(array<T> &data_, double time)
  {
    data.emplace_back();
    data.back().data.steal_data(data_);
    data.back().time = time;
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
    data.clear();
  }
};

struct CachedData {
  DataStore<Transform> transforms{};

  /* mesh data */
  DataStore<array<float3>> vertices;
  DataStore<array<int3>> triangles{};
  DataStore<array<int3>> triangles_loops{};
  DataStore<array<int>> shader{};

  /* hair data */
  DataStore<array<float3>> curve_keys;
  DataStore<array<float>> curve_radius;
  DataStore<array<int>> curve_first_key;
  DataStore<array<int>> curve_shader;

  struct CachedAttribute {
    AttributeStandard std;
    AttributeElement element;
    TypeDesc type_desc;
    ustring name;
    DataStore<array<char>> data{};
  };

  vector<CachedAttribute> attributes{};

  void clear()
  {
    vertices.clear();
    triangles.clear();
    triangles_loops.clear();
    transforms.clear();
    attributes.clear();
    curve_keys.clear();
    curve_radius.clear();
    curve_first_key.clear();
    curve_shader.clear();
    shader.clear();
  }

  CachedAttribute &add_attribute(ustring name)
  {
    for (auto &attr : attributes) {
      if (attr.name == name) {
        return attr;
      }
    }

    auto &attr = attributes.emplace_back();
    attr.name = name;
    return attr;
  }

  bool is_constant() const
  {
    if (!vertices.is_constant()) {
      return false;
    }

    if (!triangles.is_constant()) {
      return false;
    }

    if (!transforms.is_constant()) {
      return false;
    }

    if (!curve_keys.is_constant()) {
      return false;
    }

    if (!curve_radius.is_constant()) {
      return false;
    }

    if (!curve_first_key.is_constant()) {
      return false;
    }

    if (!curve_shader.is_constant()) {
      return false;
    }

    if (!shader.is_constant()) {
      return false;
    }

    for (const CachedAttribute &attr : attributes) {
      if (!attr.data.is_constant()) {
        return false;
      }
    }

    return true;
  }
};

class AlembicObject : public Node {
 public:
  NODE_DECLARE

  AlembicObject();
  ~AlembicObject();

  NODE_SOCKET_API(ustring, path)
  NODE_SOCKET_API_ARRAY(array<Node *>, used_shaders)

 private:
  friend class AlembicProcedural;

  void set_object(Object *object);
  Object *get_object();

  void load_all_data(Alembic::AbcGeom::IPolyMeshSchema &schema, Progress &progress);
  void load_all_data(const Alembic::AbcGeom::ICurvesSchema &schema, Progress &progress);

  bool has_data_loaded() const;

  MatrixSampleMap xform_samples;
  Alembic::AbcGeom::IObject iobject;
  Transform xform;

  CachedData &get_cached_data()
  {
    return cached_data;
  }

  bool is_constant() const
  {
    return cached_data.is_constant();
  }

  Object *object = nullptr;

  bool data_loaded = false;

  CachedData cached_data;

  void read_attribute(const Alembic::AbcGeom::ICompoundProperty &arb_geom_params,
                      const Alembic::AbcGeom::ISampleSelector &iss,
                      const ustring &attr_name);

  void read_face_sets(Alembic::AbcGeom::IPolyMeshSchema &schema, array<int> &polygon_to_shader);

  void setup_transform_cache();

  AttributeRequestSet get_requested_attributes();
};

class AlembicProcedural : public Procedural {
 public:
  NODE_DECLARE

  AlembicProcedural();
  ~AlembicProcedural();
  void generate(Scene *scene, Progress &progress);

  NODE_SOCKET_API(ustring, filepath)
  NODE_SOCKET_API(float, frame)
  NODE_SOCKET_API(float, frame_rate)
  NODE_SOCKET_API_ARRAY(array<Node *>, objects)

  void add_object(AlembicObject *object);

  void tag_update(Scene *scene);

 private:
  Alembic::AbcGeom::IArchive archive;
  bool objects_loaded = false;

  void load_objects(Progress &progress);

  void read_mesh(Scene *scene,
                 AlembicObject *abc_object,
                 Alembic::AbcGeom::Abc::chrono_t frame_time,
                 Progress &progress);

  void read_curves(Scene *scene,
                   AlembicObject *abc_object,
                   Alembic::AbcGeom::Abc::chrono_t frame_time,
                   Progress &progress);

  void walk_hierarchy(Alembic::AbcGeom::IObject parent,
                      const Alembic::AbcGeom::ObjectHeader &ohead,
                      MatrixSampleMap *xform_samples,
                      const unordered_map<string, AlembicObject *> &object_map,
                      Progress &progress);
};

CCL_NAMESPACE_END

#endif
