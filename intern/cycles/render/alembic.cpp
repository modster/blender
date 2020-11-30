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

#include "alembic.h"

#include <algorithm>
#include <fnmatch.h>
#include <iterator>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <vector>

#include "render/camera.h"
#include "render/curves.h"
#include "render/mesh.h"
#include "render/object.h"
#include "render/scene.h"
#include "render/shader.h"

#include "util/util_foreach.h"
#include "util/util_transform.h"
#include "util/util_vector.h"

using namespace Alembic::AbcGeom;

CCL_NAMESPACE_BEGIN

static float3 make_float3_from_yup(const Imath::Vec3<float> &v)
{
  return make_float3(v.x, -v.z, v.y);
}

static M44d convert_yup_zup(const M44d &mtx)
{
  V3d scale, shear, rotation, translation;
  extractSHRT(mtx, scale, shear, rotation, translation);

  M44d rot_mat, scale_mat, trans_mat;
  rot_mat.setEulerAngles(V3d(rotation.x, -rotation.z, rotation.y));
  scale_mat.setScale(V3d(scale.x, scale.z, scale.y));
  trans_mat.setTranslation(V3d(translation.x, -translation.z, translation.y));

  return scale_mat * rot_mat * trans_mat;
}

void transform_decompose(const Imath::M44d &mat,
                         Imath::V3d &scale,
                         Imath::V3d &shear,
                         Imath::Quatd &rotation,
                         Imath::V3d &translation)
{
  Imath::M44d mat_remainder(mat);

  /* extract scale and shear */
  Imath::extractAndRemoveScalingAndShear(mat_remainder, scale, shear);

  /* extract translation */
  translation.x = mat_remainder[3][0];
  translation.y = mat_remainder[3][1];
  translation.z = mat_remainder[3][2];

  /* extract rotation */
  rotation = extractQuat(mat_remainder);
}

M44d transform_compose(const Imath::V3d &scale,
                       const Imath::V3d &shear,
                       const Imath::Quatd &rotation,
                       const Imath::V3d &translation)
{
  Imath::M44d scale_mat, shear_mat, rot_mat, trans_mat;

  scale_mat.setScale(scale);
  shear_mat.setShear(shear);
  rot_mat = rotation.toMatrix44();
  trans_mat.setTranslation(translation);

  return scale_mat * shear_mat * rot_mat * trans_mat;
}

/* get the matrix for the specified time, or return the identity matrix if there is no exact match
 */
static M44d get_matrix_for_time(const MatrixSampleMap &samples, chrono_t time)
{
  MatrixSampleMap::const_iterator iter = samples.find(time);
  if (iter != samples.end()) {
    return iter->second;
  }

  return M44d();
}

/* get the matrix for the specified time, or interpolate between samples if there is no exact match
 */
static M44d get_interpolated_matrix_for_time(const MatrixSampleMap &samples, chrono_t time)
{
  if (samples.empty()) {
    return M44d();
  }

  /* see if exact match */
  MatrixSampleMap::const_iterator iter = samples.find(time);
  if (iter != samples.end()) {
    return iter->second;
  }

  if (samples.size() == 1) {
    return samples.begin()->second;
  }

  if (time <= samples.begin()->first) {
    return samples.begin()->second;
  }

  if (time >= samples.rbegin()->first) {
    return samples.rbegin()->second;
  }

  /* find previous and next time sample to interpolate */
  chrono_t prev_time = samples.begin()->first;
  chrono_t next_time = samples.rbegin()->first;

  for (MatrixSampleMap::const_iterator I = samples.begin(); I != samples.end(); ++I) {
    chrono_t current_time = (*I).first;

    if (current_time > prev_time && current_time <= time) {
      prev_time = current_time;
    }

    if (current_time > next_time && current_time >= time) {
      next_time = current_time;
    }
  }

  const M44d prev_mat = get_matrix_for_time(samples, prev_time);
  const M44d next_mat = get_matrix_for_time(samples, next_time);

  Imath::V3d prev_scale, next_scale;
  Imath::V3d prev_shear, next_shear;
  Imath::V3d prev_translation, next_translation;
  Imath::Quatd prev_rotation, next_rotation;

  transform_decompose(prev_mat, prev_scale, prev_shear, prev_rotation, prev_translation);
  transform_decompose(next_mat, next_scale, next_shear, next_rotation, next_translation);

  chrono_t t = (time - prev_time) / (next_time - prev_time);

  /* ensure rotation around the shortest angle  */
  if ((prev_rotation ^ next_rotation) < 0) {
    next_rotation = -next_rotation;
  }

  return transform_compose(Imath::lerp(prev_scale, next_scale, t),
                           Imath::lerp(prev_shear, next_shear, t),
                           Imath::slerp(prev_rotation, next_rotation, t),
                           Imath::lerp(prev_translation, next_translation, t));
}

static void concatenate_xform_samples(const MatrixSampleMap &parent_samples,
                                      const MatrixSampleMap &local_samples,
                                      MatrixSampleMap &output_samples)
{
  std::set<chrono_t> union_of_samples;

  for (auto &pair : parent_samples) {
    union_of_samples.insert(pair.first);
  }

  for (auto &pair : local_samples) {
    union_of_samples.insert(pair.first);
  }

  foreach (chrono_t time, union_of_samples) {
    M44d parent_matrix = get_interpolated_matrix_for_time(parent_samples, time);
    M44d local_matrix = get_interpolated_matrix_for_time(local_samples, time);

    output_samples[time] = local_matrix * parent_matrix;
  }
}

static Transform make_transform(const Abc::M44d &a)
{
  M44d m = convert_yup_zup(a);
  Transform trans;
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < 4; i++) {
      trans[j][i] = static_cast<float>(m[i][j]);
    }
  }
  return trans;
}

static void read_default_uvs(const IV2fGeomParam &uvs, CachedData &cached_data)
{
  auto &attr = cached_data.add_attribute(ustring(uvs.getName()));

  for (index_t i = 0; i < static_cast<index_t>(uvs.getNumSamples()); ++i) {
    const ISampleSelector iss = ISampleSelector(static_cast<index_t>(i));
    const IV2fGeomParam::Sample sample = uvs.getExpandedValue(iss);

    const double time = uvs.getTimeSampling()->getSampleTime(static_cast<index_t>(i));

    switch (uvs.getScope()) {
      case kFacevaryingScope: {
        IV2fGeomParam::Sample uvsample = uvs.getIndexedValue(iss);

        if (!uvsample.valid()) {
          continue;
        }

        auto triangles = cached_data.triangles.data_for_time(time);
        auto triangles_loops = cached_data.triangles_loops.data_for_time(time);

        if (!triangles || !triangles_loops) {
          continue;
        }

        attr.std = ATTR_STD_UV;

        array<char> data;
        data.resize(triangles->size() * 3 * sizeof(float2));

        float2 *data_float2 = reinterpret_cast<float2 *>(data.data());

        const unsigned int *indices = uvsample.getIndices()->get();
        const Imath::Vec2<float> *values = uvsample.getVals()->get();

        for (const int3 &loop : *triangles_loops) {
          unsigned int v0 = indices[loop.x];
          unsigned int v1 = indices[loop.y];
          unsigned int v2 = indices[loop.z];

          data_float2[0] = make_float2(values[v0][0], values[v0][1]);
          data_float2[1] = make_float2(values[v1][0], values[v1][1]);
          data_float2[2] = make_float2(values[v2][0], values[v2][1]);
          data_float2 += 3;
        }

        attr.data.add_data(data, time);

        break;
      }
      default: {
        // not supported
        break;
      }
    }
  }
}

static void read_default_normals(const IN3fGeomParam &normals,
                                 CachedData &cached_data)
{
  auto &attr = cached_data.add_attribute(ustring(normals.getName()));

  for (index_t i = 0; i < static_cast<index_t>(normals.getNumSamples()); ++i) {
    const ISampleSelector iss = ISampleSelector(static_cast<index_t>(i));
    const IN3fGeomParam::Sample sample = normals.getExpandedValue(iss);

    if (!sample.valid()) {
      return;
    }

    const double time = normals.getTimeSampling()->getSampleTime(static_cast<index_t>(i));

    switch (normals.getScope()) {
      case kFacevaryingScope: {
        attr.std = ATTR_STD_VERTEX_NORMAL;

        auto vertices = cached_data.vertices.data_for_time(time);
        auto triangles = cached_data.triangles.data_for_time(time);

        if (!vertices || !triangles) {
          continue;
        }

        array<char> data;
        data.resize(vertices->size() * sizeof(float3));

        float3 *data_float3 = reinterpret_cast<float3 *>(data.data());

        for (size_t i = 0; i < vertices->size(); ++i) {
          data_float3[i] = make_float3(0.0f);
        }

        const Imath::V3f *values = sample.getVals()->get();

        for (const int3 &tri : *triangles) {
          const Imath::V3f &v0 = values[tri.x];
          const Imath::V3f &v1 = values[tri.y];
          const Imath::V3f &v2 = values[tri.z];

          data_float3[tri.x] += make_float3_from_yup(v0);
          data_float3[tri.y] += make_float3_from_yup(v1);
          data_float3[tri.z] += make_float3_from_yup(v2);
        }

        attr.data.add_data(data, time);

        break;
      }
      case kVaryingScope:
      case kVertexScope: {
        attr.std = ATTR_STD_VERTEX_NORMAL;

        auto vertices = cached_data.vertices.data_for_time(time);

        if (!vertices) {
          continue;
        }

        array<char> data;
        data.resize(vertices->size() * sizeof(float3));

        float3 *data_float3 = reinterpret_cast<float3 *>(data.data());

        const Imath::V3f *values = sample.getVals()->get();

        for (size_t i = 0; i < vertices->size(); ++i) {
          data_float3[i] = make_float3_from_yup(values[i]);
        }

        attr.data.add_data(data, time);

        break;
      }
      default: {
        // not supported
        break;
      }
    }
  }
}

NODE_DEFINE(AlembicObject)
{
  NodeType *type = NodeType::add("alembic_object", create);
  SOCKET_STRING(path, "Alembic Path", ustring());
  SOCKET_NODE_ARRAY(used_shaders, "Used Shaders", &Shader::node_type);

  return type;
}

AlembicObject::AlembicObject() : Node(node_type)
{
}

AlembicObject::~AlembicObject()
{
}

void AlembicObject::set_object(Object *object_)
{
  object = object_;
}

Object *AlembicObject::get_object()
{
  return object;
}

bool AlembicObject::has_data_loaded() const
{
  return data_loaded;
}

void AlembicObject::load_all_data(const IPolyMeshSchema &schema)
{
  cached_data.clear();

  Geometry *geometry = object->get_geometry();
  assert(geometry);

  AttributeRequestSet requested_attributes;

  // TODO : check for attribute changes in the shaders
  foreach (Node *node, geometry->get_used_shaders()) {
    Shader *shader = static_cast<Shader *>(node);

    foreach (const AttributeRequest &attr, shader->attributes.requests) {
      if (attr.name != "") {
        requested_attributes.add(attr.name);
      }
    }
  }

  cached_data.vertices.set_time_sampling(*schema.getTimeSampling());
  cached_data.triangles.set_time_sampling(*schema.getTimeSampling());
  cached_data.triangles_loops.set_time_sampling(*schema.getTimeSampling());

  for (size_t i = 0; i < schema.getNumSamples(); ++i) {
    const ISampleSelector iss = ISampleSelector(static_cast<index_t>(i));
    const IPolyMeshSchema::Sample sample = schema.getValue(iss);

    const P3fArraySamplePtr positions = sample.getPositions();

    const double time = schema.getTimeSampling()->getSampleTime(static_cast<index_t>(i));

    if (positions) {
      array<float3> vertices;
      vertices.reserve(positions->size());

      for (int i = 0; i < positions->size(); i++) {
        Imath::Vec3<float> f = positions->get()[i];
        vertices.push_back_reserved(make_float3_from_yup(f));
      }

      cached_data.vertices.add_data(vertices, time);
    }

    Int32ArraySamplePtr face_counts = sample.getFaceCounts();
    Int32ArraySamplePtr face_indices = sample.getFaceIndices();

    if (face_counts && face_indices) {
      const size_t num_faces = face_counts->size();
      const int *face_counts_array = face_counts->get();
      const int *face_indices_array = face_indices->get();

      size_t num_triangles = 0;
      for (size_t i = 0; i < face_counts->size(); i++) {
        num_triangles += face_counts_array[i] - 2;
      }

      array<int3> triangles;
      array<int3> triangles_loops;
      triangles.reserve(num_triangles);
      triangles_loops.reserve(num_triangles);
      int index_offset = 0;

      for (size_t i = 0; i < num_faces; i++) {
        for (int j = 0; j < face_counts_array[i] - 2; j++) {
          int v0 = face_indices_array[index_offset];
          int v1 = face_indices_array[index_offset + j + 1];
          int v2 = face_indices_array[index_offset + j + 2];

          triangles.push_back_reserved(make_int3(v0, v1, v2));
          triangles_loops.push_back_reserved(
              make_int3(index_offset, index_offset + j + 1, index_offset + j + 2));
        }

        index_offset += face_counts_array[i];
      }

      cached_data.triangles.add_data(triangles, time);
      cached_data.triangles_loops.add_data(triangles_loops, time);
    }

    foreach (const AttributeRequest &attr, requested_attributes.requests) {
      read_attribute(schema.getArbGeomParams(), iss, attr.name);
    }
  }

  const IV2fGeomParam &uvs = schema.getUVsParam();

  if (uvs.valid()) {
    read_default_uvs(uvs, cached_data);
  }

  //  const IN3fGeomParam &normals = schema.getNormalsParam();

  //  if (normals.valid()) {
  //    read_default_normals(normals, cached_data);
  //  }

  if (xform_samples.size() == 0) {
    cached_data.transforms.add_data(transform_identity(), 0.0);
  }
  else {
    for (auto &pair : xform_samples) {
      Transform tfm = make_transform(pair.second);
      cached_data.transforms.add_data(tfm, pair.first);
    }
  }

  // TODO : proper time sampling, but is it possible for the hierarchy to have different time
  // sampling for each xform ?
  cached_data.transforms.set_time_sampling(cached_data.vertices.time_sampling);

  data_loaded = true;
}

void AlembicObject::read_attribute(const ICompoundProperty &arb_geom_params,
                                   const ISampleSelector &iss,
                                   const ustring &attr_name)
{
  auto index = iss.getRequestedIndex();
  auto &attribute = cached_data.add_attribute(attr_name);

  for (size_t i = 0; i < arb_geom_params.getNumProperties(); ++i) {
    const PropertyHeader &prop = arb_geom_params.getPropertyHeader(i);

    if (prop.getName() != attr_name) {
      continue;
    }

    if (IV2fProperty::matches(prop.getMetaData()) && Alembic::AbcGeom::isUV(prop)) {
      const IV2fGeomParam &param = IV2fGeomParam(arb_geom_params, prop.getName());

      IV2fGeomParam::Sample sample;
      param.getIndexed(sample, iss);

      auto time = param.getTimeSampling()->getSampleTime(index);

      if (param.getScope() == kFacevaryingScope) {
        V2fArraySamplePtr values = sample.getVals();
        UInt32ArraySamplePtr indices = sample.getIndices();

        attribute.std = ATTR_STD_NONE;
        attribute.element = ATTR_ELEMENT_CORNER;
        attribute.type_desc = TypeFloat2;

        auto triangles = cached_data.triangles.data_for_time(time);
        auto triangles_loops = cached_data.triangles_loops.data_for_time(time);

        if (!triangles || !triangles_loops) {
          continue;
        }

        array<char> data;
        data.resize(triangles->size() * 3 * sizeof(float2));

        float2 *data_float2 = reinterpret_cast<float2 *>(data.data());

        for (const int3 &loop : *triangles_loops) {
          unsigned int v0 = (*indices)[loop.x];
          unsigned int v1 = (*indices)[loop.y];
          unsigned int v2 = (*indices)[loop.z];

          data_float2[0] = make_float2((*values)[v0][0], (*values)[v0][1]);
          data_float2[1] = make_float2((*values)[v1][0], (*values)[v1][1]);
          data_float2[2] = make_float2((*values)[v2][0], (*values)[v2][1]);
          data_float2 += 3;
        }

        attribute.data.set_time_sampling(*param.getTimeSampling());
        attribute.data.add_data(data, time);
      }
    }
    else if (IC3fProperty::matches(prop.getMetaData())) {
      const IC3fGeomParam &param = IC3fGeomParam(arb_geom_params, prop.getName());

      IC3fGeomParam::Sample sample;
      param.getIndexed(sample, iss);

      auto time = param.getTimeSampling()->getSampleTime(index);

      C3fArraySamplePtr values = sample.getVals();

      attribute.std = ATTR_STD_NONE;

      if (param.getScope() == kVaryingScope) {
        attribute.element = ATTR_ELEMENT_CORNER_BYTE;
        attribute.type_desc = TypeRGBA;

        auto triangles = cached_data.triangles.data_for_time(time);

        if (!triangles) {
          continue;
        }

        array<char> data;
        data.resize(triangles->size() * 3 * sizeof(uchar4));

        uchar4 *data_uchar4 = reinterpret_cast<uchar4 *>(data.data());

        int offset = 0;
        for (const int3 &tri : *triangles) {
          Imath::C3f v = (*values)[tri.x];
          data_uchar4[offset + 0] = color_float_to_byte(make_float3(v.x, v.y, v.z));

          v = (*values)[tri.y];
          data_uchar4[offset + 1] = color_float_to_byte(make_float3(v.x, v.y, v.z));

          v = (*values)[tri.z];
          data_uchar4[offset + 2] = color_float_to_byte(make_float3(v.x, v.y, v.z));

          offset += 3;
        }

        attribute.data.set_time_sampling(*param.getTimeSampling());
        attribute.data.add_data(data, time);
      }
    }
    else if (IC4fProperty::matches(prop.getMetaData())) {
      const IC4fGeomParam &param = IC4fGeomParam(arb_geom_params, prop.getName());

      IC4fGeomParam::Sample sample;
      param.getIndexed(sample, iss);

      auto time = param.getTimeSampling()->getSampleTime(index);

      C4fArraySamplePtr values = sample.getVals();

      attribute.std = ATTR_STD_NONE;

      if (param.getScope() == kVaryingScope) {
        attribute.element = ATTR_ELEMENT_CORNER_BYTE;
        attribute.type_desc = TypeRGBA;

        auto triangles = cached_data.triangles.data_for_time(time);

        if (!triangles) {
          continue;
        }

        array<char> data;
        data.resize(triangles->size() * 3 * sizeof(uchar4));

        uchar4 *data_uchar4 = reinterpret_cast<uchar4 *>(data.data());

        int offset = 0;
        for (const int3 &tri : *triangles) {
          Imath::C4f v = (*values)[tri.x];
          data_uchar4[offset + 0] = color_float4_to_uchar4(make_float4(v.r, v.g, v.b, v.a));

          v = (*values)[tri.y];
          data_uchar4[offset + 1] = color_float4_to_uchar4(make_float4(v.r, v.g, v.b, v.a));

          v = (*values)[tri.z];
          data_uchar4[offset + 2] = color_float4_to_uchar4(make_float4(v.r, v.g, v.b, v.a));

          offset += 3;
        }

        attribute.data.set_time_sampling(*param.getTimeSampling());
        attribute.data.add_data(data, time);
      }
    }
  }
}

NODE_DEFINE(AlembicProcedural)
{
  NodeType *type = NodeType::add("alembic", create);

  SOCKET_BOOLEAN(use_motion_blur, "Use Motion Blur", false);

  SOCKET_STRING(filepath, "Filename", ustring());
  SOCKET_FLOAT(frame, "Frame", 1.0f);
  SOCKET_FLOAT(frame_rate, "Frame Rate", 24.0f);

  SOCKET_NODE_ARRAY(objects, "Objects", &AlembicObject::node_type);

  return type;
}

AlembicProcedural::AlembicProcedural() : Procedural(node_type)
{
  frame = 1.0f;
  frame_rate = 24.0f;
}

AlembicProcedural::~AlembicProcedural()
{
  for (size_t i = 0; i < objects.size(); ++i) {
    delete objects[i];
  }
}

void AlembicProcedural::generate(Scene *scene)
{
  if (!is_modified()) {
    return;
  }

  if (!archive.valid()) {
    Alembic::AbcCoreFactory::IFactory factory;
    factory.setPolicy(Alembic::Abc::ErrorHandler::kQuietNoopPolicy);
    archive = factory.getArchive(filepath.c_str());

    if (!archive.valid()) {
      /* avoid potential infinite update loops in viewport synchronization */
      filepath.clear();
      clear_modified();
      return;
    }
  }

  if (!objects_loaded) {
    load_objects();
    objects_loaded = true;
  }

  Abc::chrono_t frame_time = (Abc::chrono_t)(frame / frame_rate);

  int objects_updated = 0;

  foreach (AlembicObject *object, objects) {
    /* skip constant objects */
    if (object->has_data_loaded() && object->is_constant()) {
      continue;
    }

    if (IPolyMesh::matches(object->iobject.getHeader())) {
      IPolyMesh mesh(object->iobject, Alembic::Abc::kWrapExisting);
      read_mesh(scene, object, object->xform, mesh, frame_time);
    }
    else if (ICurves::matches(object->iobject.getHeader())) {
      ICurves curves(object->iobject, Alembic::Abc::kWrapExisting);
      read_curves(scene, object, object->xform, curves, frame_time);
    }

    objects_updated += 1;
  }

  clear_modified();
}

void AlembicProcedural::tag_update(Scene *scene)
{
  if (is_modified()) {
    scene->procedural_manager->need_update = true;
  }
}

void AlembicProcedural::load_objects()
{
  unordered_map<string, AlembicObject *> object_map;

  foreach (AlembicObject *object, objects) {
    object_map.insert({object->get_path().c_str(), object});
  }

  IObject root = archive.getTop();

  for (size_t i = 0; i < root.getNumChildren(); ++i) {
    walk_hierarchy(root, root.getChildHeader(i), nullptr, object_map);
  }
}

void AlembicProcedural::read_mesh(Scene *scene,
                                  AlembicObject *abc_object,
                                  Transform xform,
                                  IPolyMesh &polymesh,
                                  Abc::chrono_t frame_time)
{
  Mesh *mesh = nullptr;

  /* create a mesh node in the scene if not already done */
  if (!abc_object->get_object()) {
    mesh = scene->create_node<Mesh>();
    mesh->set_use_motion_blur(use_motion_blur);
    mesh->name = abc_object->iobject.getName();

    array<Node *> used_shaders = abc_object->get_used_shaders();
    mesh->set_used_shaders(used_shaders);

    /* create object*/
    Object *object = scene->create_node<Object>();
    object->set_geometry(mesh);
    object->set_tfm(xform);
    object->name = abc_object->iobject.getName();

    abc_object->set_object(object);
  }
  else {
    mesh = static_cast<Mesh *>(abc_object->get_object()->get_geometry());
  }

  IPolyMeshSchema schema = polymesh.getSchema();

  if (!abc_object->has_data_loaded()) {
    abc_object->load_all_data(schema);
  }

  auto &cached_data = abc_object->get_cached_data();

  // TODO : arrays are emptied when passed to the sockets, so we need to reload the data
  // perhaps we should just have a way to set the pointer
  if (cached_data.is_dirty_frame(frame_time)) {
    abc_object->load_all_data(schema);
  }

  Transform *tfm = cached_data.transforms.data_for_time(frame_time);
  if (tfm) {
    auto object = abc_object->get_object();
    object->set_tfm(*tfm);
  }

  array<float3> *vertices = cached_data.vertices.data_for_new_time(frame_time);
  if (vertices) {
    cached_data.add_dirty_frame(frame_time);
    mesh->set_verts(*vertices);
  }

  array<int3> *triangle_data = cached_data.triangles.data_for_new_time(frame_time);
  if (triangle_data) {
    cached_data.add_dirty_frame(frame_time);
    // TODO : shader association
    array<int> triangles;
    array<bool> smooth;
    array<int> shader;

    triangles.reserve(triangle_data->size() * 3);
    smooth.reserve(triangle_data->size());
    shader.reserve(triangle_data->size());

    for (int i = 0; i < triangle_data->size(); ++i) {
      int3 tri = (*triangle_data)[i];
      triangles.push_back_reserved(tri.x);
      triangles.push_back_reserved(tri.y);
      triangles.push_back_reserved(tri.z);
      shader.push_back_reserved(0);
      smooth.push_back_reserved(1);
    }

    mesh->set_triangles(triangles);
    mesh->set_smooth(smooth);
    mesh->set_shader(shader);
  }

  /* we don't yet support arbitrary attributes, for now add vertex
   * coordinates as generated coordinates if requested */
  if (mesh->need_attribute(scene, ATTR_STD_GENERATED)) {
    Attribute *attr = mesh->attributes.add(ATTR_STD_GENERATED);
    memcpy(
        attr->data_float3(), mesh->get_verts().data(), sizeof(float3) * mesh->get_verts().size());
  }

  for (auto &attribute : cached_data.attributes) {
    auto attr_data = attribute.data.data_for_new_time(frame_time);

    if (!attr_data) {
      continue;
    }

    Attribute *attr = nullptr;
    if (attribute.std != ATTR_STD_NONE) {
      attr = mesh->attributes.add(attribute.std, attribute.name);
    }
    else {
      attr = mesh->attributes.add(attribute.name, attribute.type_desc, attribute.element);
    }
    assert(attr);

    memcpy(attr->data(), attr_data->data(), attr_data->size());
  }

  /* TODO: read normals from the archive if present */
  mesh->add_face_normals();

  if (mesh->is_modified()) {
    // TODO : check for modification of subdivision data (is a separate object in Alembic)
    bool need_rebuild = mesh->triangles_is_modified();
    mesh->tag_update(scene, need_rebuild);
  }
}

void AlembicProcedural::read_curves(Scene *scene,
                                    AlembicObject *abc_object,
                                    Transform xform,
                                    ICurves &curves,
                                    Abc::chrono_t frame_time)
{
  Hair *hair;

  /* create a hair node in the scene if not already done */
  if (!abc_object->get_object()) {
    hair = scene->create_node<Hair>();
    hair->set_use_motion_blur(use_motion_blur);
    hair->name = abc_object->iobject.getName();

    array<Node *> used_shaders = abc_object->get_used_shaders();
    hair->set_used_shaders(used_shaders);

    /* create object*/
    Object *object = scene->create_node<Object>();
    object->set_geometry(hair);
    object->set_tfm(xform);
    object->name = abc_object->iobject.getName();

    abc_object->set_object(object);
  }
  else {
    hair = static_cast<Hair *>(abc_object->get_object()->get_geometry());
  }

  ICurvesSchema::Sample samp = curves.getSchema().getValue(ISampleSelector(frame_time));

  hair->clear();
  hair->reserve_curves(samp.getNumCurves(), samp.getPositions()->size());

  Abc::Int32ArraySamplePtr curveNumVerts = samp.getCurvesNumVertices();
  int offset = 0;
  for (int i = 0; i < curveNumVerts->size(); i++) {
    int numVerts = curveNumVerts->get()[i];
    for (int j = 0; j < numVerts; j++) {
      Imath::Vec3<float> f = samp.getPositions()->get()[offset + j];
      hair->add_curve_key(make_float3_from_yup(f), 0.01f);
    }
    hair->add_curve(offset, 0);
    offset += numVerts;
  }

  if (use_motion_blur) {
    Attribute *attr = hair->attributes.add(ATTR_STD_MOTION_VERTEX_POSITION);
    float3 *fdata = attr->data_float3();
    float shuttertimes[2] = {-scene->camera->get_shuttertime() / 2.0f,
                             scene->camera->get_shuttertime() / 2.0f};
    AbcA::TimeSamplingPtr ts = curves.getSchema().getTimeSampling();
    for (int i = 0; i < 2; i++) {
      frame_time = static_cast<Abc::chrono_t>((frame + shuttertimes[i]) / frame_rate);
      std::pair<index_t, chrono_t> idx = ts->getNearIndex(frame_time,
                                                          curves.getSchema().getNumSamples());
      ICurvesSchema::Sample shuttersamp = curves.getSchema().getValue(idx.first);
      for (int i = 0; i < shuttersamp.getPositions()->size(); i++) {
        Imath::Vec3<float> f = shuttersamp.getPositions()->get()[i];
        float3 p = make_float3_from_yup(f);
        *fdata++ = p;
      }
    }
  }

  /* we don't yet support arbitrary attributes, for now add vertex
   * coordinates as generated coordinates if requested */
  if (hair->need_attribute(scene, ATTR_STD_GENERATED)) {
    // TODO : add generated coordinates for curves
  }
}

void AlembicProcedural::walk_hierarchy(
    IObject parent,
    const ObjectHeader &header,
    MatrixSampleMap *xform_samples,
    const unordered_map<std::string, AlembicObject *> &object_map)
{
  IObject next_object;

  MatrixSampleMap concatenated_xform_samples;

  if (IXform::matches(header)) {
    IXform xform(parent, header.getName());

    IXformSchema &xs = xform.getSchema();

    if (xs.getNumOps() > 0) {
      TimeSamplingPtr ts = xs.getTimeSampling();
      MatrixSampleMap local_xform_samples;

      MatrixSampleMap *temp_xform_samples = nullptr;
      if (xform_samples == nullptr) {
        /* If there is no parent transforms, fill the map directly. */
        temp_xform_samples = &concatenated_xform_samples;
      }
      else {
        /* use a temporary map */
        temp_xform_samples = &local_xform_samples;
      }

      for (size_t i = 0; i < xs.getNumSamples(); ++i) {
        chrono_t sample_time = ts->getSampleTime(i);
        XformSample sample = xs.getValue(ISampleSelector(sample_time));
        temp_xform_samples->insert({sample_time, sample.getMatrix()});
      }

      if (xform_samples != nullptr) {
        concatenate_xform_samples(*xform_samples, local_xform_samples, concatenated_xform_samples);
      }

      xform_samples = &concatenated_xform_samples;
    }

    next_object = xform;
  }
  else if (ISubD::matches(header)) {
    // todo: subdivision
  }
  else if (IPolyMesh::matches(header)) {
    IPolyMesh mesh(parent, header.getName());

    auto iter = object_map.find(mesh.getFullName());

    if (iter != object_map.end()) {
      AlembicObject *abc_object = iter->second;
      abc_object->iobject = mesh;

      if (xform_samples) {
        abc_object->xform_samples = *xform_samples;
      }
    }

    next_object = mesh;
  }
  else if (ICurves::matches(header)) {
    ICurves curves(parent, header.getName());

    auto iter = object_map.find(curves.getFullName());

    if (iter != object_map.end()) {
      AlembicObject *abc_object = iter->second;
      abc_object->iobject = curves;

      if (xform_samples) {
        abc_object->xform_samples = *xform_samples;
      }
    }

    next_object = curves;
  }
  else if (IFaceSet::matches(header)) {
    // ignore the face set, it will be read along with the data
  }
  else {
    // unsupported type for now (Points, NuPatch)
    next_object = parent.getChild(header.getName());
  }

  if (next_object.valid()) {
    for (size_t i = 0; i < next_object.getNumChildren(); ++i) {
      walk_hierarchy(next_object, next_object.getChildHeader(i), xform_samples, object_map);
    }
  }
}

CCL_NAMESPACE_END
