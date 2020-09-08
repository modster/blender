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

CCL_NAMESPACE_BEGIN

static float3 make_float3_from_yup(Imath::Vec3<float> const &v)
{
  return make_float3(v.x, -v.z, v.y);
}

static M44d convert_yup_zup(M44d const &mtx)
{
  Imath::Vec3<double> scale, shear, rot, trans;
  extractSHRT(mtx, scale, shear, rot, trans);
  M44d rotmat, scalemat, transmat;
  rotmat.setEulerAngles(Imath::Vec3<double>(rot.x, -rot.z, rot.y));
  scalemat.setScale(Imath::Vec3<double>(scale.x, scale.z, scale.y));
  transmat.setTranslation(Imath::Vec3<double>(trans.x, -trans.z, trans.y));
  return scalemat * rotmat * transmat;
}

static Transform make_transform(const Abc::M44d &a)
{
  auto m = convert_yup_zup(a);
  Transform trans;
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < 4; i++) {
      trans[j][i] = static_cast<float>(m[i][j]);
    }
  }
  return trans;
}

/* TODO: any attribute lookup should probably go through the AttributeRequests
 */
static void read_uvs(const IV2fGeomParam &uvs, Geometry *node, const int *face_counts, const int num_faces)
{
  if (uvs.valid()) {
    switch (uvs.getScope()) {
      case kVaryingScope:
      case kVertexScope:
      {
        IV2fGeomParam::Sample uvsample = uvs.getExpandedValue();
        break;
      }
      case kFacevaryingScope:
      {
        IV2fGeomParam::Sample uvsample = uvs.getIndexedValue();

        ustring name = ustring("UVMap");
        Attribute *attr = node->attributes.add(ATTR_STD_UV, name);
        float2 *fdata = attr->data_float2();

        /* loop over the triangles */
        int index_offset = 0;
        const unsigned int *uvIndices = uvsample.getIndices()->get();
        const Imath::Vec2<float> *uvValues = uvsample.getVals()->get();

        for (size_t i = 0; i < num_faces; i++) {
          for (int j = 0; j < face_counts[i] - 2; j++) {
            int v0 = uvIndices[index_offset];
            int v1 = uvIndices[index_offset + j + 1];
            int v2 = uvIndices[index_offset + j + 2];

            fdata[0] = make_float2(uvValues[v0][0], uvValues[v0][1]);
            fdata[1] = make_float2(uvValues[v1][0], uvValues[v1][1]);
            fdata[2] = make_float2(uvValues[v2][0], uvValues[v2][1]);
            fdata += 3;
          }

          index_offset += face_counts[i];
        }

        break;
      }
      default:
      {
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

int AlembicObject::frame_index(float frame, float frame_rate)
{
  return static_cast<int>(frame - (static_cast<float>(min_time) * frame_rate));
}

AlembicObject::DataCache &AlembicObject::get_frame_data(int index)
{
  if (index < 0) {
    return frame_data[0];
  }

  if (index >= frame_data.size()) {
    return frame_data.back();
  }

  return frame_data[index];
}

void AlembicObject::load_all_data(IPolyMeshSchema &schema)
{
  frame_data.clear();

  const auto &time_sampling = schema.getTimeSampling();

  if (!schema.isConstant()) {
    auto num_samples = schema.getNumSamples();

    if (num_samples > 0) {
      min_time = time_sampling->getSampleTime(0);
      max_time = time_sampling->getSampleTime(num_samples - 1);
    }
  }

  // TODO : store other properties and have a better structure to store these arrays
  for (size_t i = 0; i < schema.getNumSamples(); ++i) {
    frame_data.emplace_back();
    AlembicObject::DataCache &data_cache = frame_data.back();

    auto sample = schema.getValue(ISampleSelector(static_cast<index_t>(i)));

    auto positions = sample.getPositions();

    if (positions) {
      data_cache.vertices.reserve(positions->size());

      for (int i = 0; i < positions->size(); i++) {
        Imath::Vec3<float> f = positions->get()[i];
        data_cache.vertices.push_back_reserved(make_float3_from_yup(f));
      }
    }

    auto face_counts = sample.getFaceCounts();
    auto face_indices = sample.getFaceIndices();

    if (face_counts && face_indices) {
      int num_faces = face_counts->size();
      const int *face_counts_array = face_counts->get();
      const int *face_indices_array = face_indices->get();

      int num_triangles = 0;
      for (int i = 0; i < face_counts->size(); i++) {
        num_triangles += face_counts_array[i] - 2;
      }

      data_cache.triangles.reserve(num_triangles);
      int index_offset = 0;

      for (size_t i = 0; i < num_faces; i++) {
        for (int j = 0; j < face_counts_array[i] - 2; j++) {
          int v0 = face_indices_array[index_offset];
          int v1 = face_indices_array[index_offset + j + 1];
          int v2 = face_indices_array[index_offset + j + 2];

          data_cache.triangles.push_back_reserved(make_int3(v0, v1, v2));
        }

        index_offset += face_counts_array[i];
      }
    }
  }

  data_loaded = true;
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
  for (auto i = 0; i < objects.size(); ++i) {
    delete objects[i];
  }
}

void AlembicProcedural::generate(Scene *scene)
{
  if (!is_modified()) {
    return;
  }

  Alembic::AbcCoreFactory::IFactory factory;
  factory.setPolicy(Alembic::Abc::ErrorHandler::kQuietNoopPolicy);
  IArchive archive = factory.getArchive(filepath.c_str());

  if (!archive.valid()) {
    // TODO : error reporting
    return;
  }

  auto frame_time = (Abc::chrono_t)(frame / frame_rate);

  /* Traverse Alembic file hierarchy, avoiding recursion by
   * using an explicit stack
   *
   * TODO : cache this traversal
   */
  std::stack<std::pair<IObject, Transform>> objstack;
  objstack.push(std::pair<IObject, Transform>(archive.getTop(), transform_identity()));

  while (!objstack.empty()) {
    std::pair<IObject, Transform> obj = objstack.top();
    objstack.pop();

    string path = obj.first.getFullName();
    Transform currmatrix = obj.second;

    AlembicObject *object = NULL;

    for (int i = 0; i < objects.size(); i++) {
      if (fnmatch(objects[i]->path.c_str(), path.c_str(), 0) == 0) {
        object = objects[i];
      }
    }

    if (IXform::matches(obj.first.getHeader())) {
      IXform xform(obj.first, Alembic::Abc::kWrapExisting);
      XformSample samp = xform.getSchema().getValue(ISampleSelector(frame_time));
      Transform ax = make_transform(samp.getMatrix());
      currmatrix = currmatrix * ax;
    }
    else if (IPolyMesh::matches(obj.first.getHeader()) && object) {
      IPolyMesh mesh(obj.first, Alembic::Abc::kWrapExisting);
      read_mesh(scene, object, currmatrix, mesh, frame_time);
    }
    else if (ICurves::matches(obj.first.getHeader()) && object) {
      ICurves curves(obj.first, Alembic::Abc::kWrapExisting);
      read_curves(scene, object, currmatrix, curves, frame_time);
    }

    for (int i = 0; i < obj.first.getNumChildren(); i++)
      objstack.push(std::pair<IObject, Transform>(obj.first.getChild(i), currmatrix));
  }

  clear_modified();
}

void AlembicProcedural::read_mesh(Scene *scene,
                                  AlembicObject *abc_object,
                                  Transform xform,
                                  IPolyMesh &polymesh,
                                  Abc::chrono_t frame_time)
{
  // TODO : support animation at the transformation level
  Mesh *mesh = nullptr;

  /* create a mesh node in the scene if not already done */
  if (!abc_object->get_object()) {
    mesh = scene->create_node<Mesh>();
    mesh->set_use_motion_blur(use_motion_blur);

    array<Shader *> used_shaders = abc_object->get_used_shaders();
    mesh->set_used_shaders(used_shaders);

    /* create object*/
    Object *object = scene->create_node<Object>();
    object->set_geometry(mesh);
    object->tfm = xform;

    abc_object->set_object(object);
  }
  else {
    mesh = static_cast<Mesh *>(abc_object->get_object()->get_geometry());
  }

  // TODO : properly check if and what data needs to be rebuild
  if (mesh->get_time_stamp() == static_cast<int>(frame)) {
    return;
  }

  mesh->set_time_stamp(static_cast<int>(frame));

  IPolyMeshSchema schema = polymesh.getSchema();

  if (!abc_object->has_data_loaded()) {
    abc_object->load_all_data(schema);
  }

  int frame_index = abc_object->frame_index(frame, frame_rate);
  auto &data = abc_object->get_frame_data(frame_index);

  // TODO : arrays are emptied when passed to the sockets, so we need to reload the data
  // perhaps we should just have a way to set the pointer
  if (data.dirty) {
    abc_object->load_all_data(schema);
    data = abc_object->get_frame_data(frame_index);
  }

  data.dirty = true;
  // TODO : animations like fluids will have different data on different frames
  auto &triangle_data = abc_object->get_frame_data(0).triangles;

  mesh->set_verts(data.vertices);

  {
    // TODO : shader association
    array<int> triangles;
    array<bool> smooth;
    array<int> shader;

    triangles.reserve(triangle_data.size() * 3);
    smooth.reserve(triangle_data.size());
    shader.reserve(triangle_data.size());

    for (int i = 0; i < triangle_data.size(); ++i) {
      int3 tri = triangle_data[i];
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

  IPolyMeshSchema::Sample samp = schema.getValue(ISampleSelector(frame_time));
  IV2fGeomParam uvs = polymesh.getSchema().getUVsParam();
  read_uvs(uvs, mesh, samp.getFaceCounts()->get(), samp.getFaceCounts()->size());

  /* TODO: read normals from the archive if present */
  mesh->add_face_normals();

  /* we don't yet support arbitrary attributes, for now add vertex
   * coordinates as generated coordinates if requested */
  if (mesh->need_attribute(scene, ATTR_STD_GENERATED)) {
    Attribute *attr = mesh->attributes.add(ATTR_STD_GENERATED);
    memcpy(
        attr->data_float3(), mesh->get_verts().data(), sizeof(float3) * mesh->get_verts().size());
  }

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
  // TODO : support animation at the transformation level
  Hair *hair;

  /* create a hair node in the scene if not already done */
  if (!abc_object->get_object()) {
    hair = scene->create_node<Hair>();
    hair->set_use_motion_blur(use_motion_blur);

    array<Shader *> used_shaders = abc_object->get_used_shaders();
    hair->set_used_shaders(used_shaders);

    /* create object*/
    Object *object = scene->create_node<Object>();
    object->geometry = hair;
    object->tfm = xform;

    abc_object->set_object(object);
  }
  else {
    hair = static_cast<Hair *>(abc_object->get_object()->get_geometry());
  }

  ICurvesSchema::Sample samp = curves.getSchema().getValue(ISampleSelector(frame_time));

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
    float shuttertimes[2] = {-scene->camera->shuttertime / 2.0f,
                             scene->camera->shuttertime / 2.0f};
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

CCL_NAMESPACE_END
