#include "BKE_cloth_remesh.hh"

#include "BKE_customdata.h"
#include "BKE_idtype.h"
#include "BKE_main.h"
#include "BLI_float3.hh"
#include "DNA_customdata_types.h"
#include "DNA_meshdata_types.h"
#include "bmesh.h"

#include "BKE_mesh.h"

#include "DNA_mesh_types.h"

#include "bmesh_class.h"
#include "intern/bmesh_construct.h"
#include "intern/bmesh_core.h"
#include "intern/bmesh_iterators.h"
#include "intern/bmesh_mesh.h"
#include "intern/bmesh_mesh_convert.h"
#include "intern/bmesh_operators.h"
#include "testing/testing.h"

#include <gtest/gtest.h>
#include <sstream>

namespace blender::bke::tests {

static std::string stringify_v2(const float *v2)
{
  std::ostringstream stream;
  stream << "(" << v2[0] << ", " << v2[1] << ")";
  return stream.str();
}

static std::string stringify_v3(const float *v3)
{
  std::ostringstream stream;
  stream << "(" << v3[0] << ", " << v3[1] << ", " << v3[2] << ")";
  return stream.str();
}

using namespace internal;

static const char *cube_pos_uv_normal =
    "# Blender v3.0.0 Alpha OBJ File: ''\n"
    "# www.blender.org\n"
    "mtllib untitled.mtl\n"
    "o Cube\n"
    "v 1.000000 1.000000 -1.000000\n"
    "v 1.000000 -1.000000 -1.000000\n"
    "v 1.000000 1.000000 1.000000\n"
    "v 1.000000 -1.000000 1.000000\n"
    "v -1.000000 1.000000 -1.000000\n"
    "v -1.000000 -1.000000 -1.000000\n"
    "v -1.000000 1.000000 1.000000\n"
    "v -1.000000 -1.000000 1.000000\n"
    "vt 0.625000 0.500000\n"
    "vt 0.875000 0.500000\n"
    "vt 0.875000 0.750000\n"
    "vt 0.625000 0.750000\n"
    "vt 0.375000 0.750000\n"
    "vt 0.625000 1.000000\n"
    "vt 0.375000 1.000000\n"
    "vt 0.375000 0.000000\n"
    "vt 0.625000 0.000000\n"
    "vt 0.625000 0.250000\n"
    "vt 0.375000 0.250000\n"
    "vt 0.125000 0.500000\n"
    "vt 0.375000 0.500000\n"
    "vt 0.125000 0.750000\n"
    "vn 0.0000 1.0000 0.0000\n"
    "vn 0.0000 0.0000 1.0000\n"
    "vn -1.0000 0.0000 0.0000\n"
    "vn 0.0000 -1.0000 0.0000\n"
    "vn 1.0000 0.0000 0.0000\n"
    "vn 0.0000 0.0000 -1.0000\n"
    "usemtl Material\n"
    "s off\n"
    "f 1/1/1 5/2/1 7/3/1 3/4/1\n"
    "f 4/5/2 3/4/2 7/6/2 8/7/2\n"
    "f 8/8/3 7/9/3 5/10/3 6/11/3\n"
    "f 6/12/4 2/13/4 4/5/4 8/14/4\n"
    "f 2/13/5 1/1/5 3/4/5 4/5/5\n"
    "f 6/11/6 5/10/6 1/1/6 2/13/6\n";

static const char *plane_extra_loose_edges =
    "# Blender v3.0.0 Alpha OBJ File: ''\n"
    "# www.blender.org\n"
    "mtllib plane_extra_loose_edges.mtl\n"
    "o Plane\n"
    "v -1.000000 0.000000 1.000000\n"
    "v 1.000000 0.000000 1.000000\n"
    "v -1.000000 0.000000 -1.000000\n"
    "v 1.000000 0.000000 -1.000000\n"
    "v -1.000000 0.000000 -2.000000\n"
    "v 1.000000 0.000000 -2.000000\n"
    "v 2.000000 0.000000 1.000000\n"
    "v 2.000000 0.000000 -1.000000\n"
    "v -1.000000 0.000000 2.000000\n"
    "v 1.000000 0.000000 2.000000\n"
    "v -2.000000 0.000000 1.000000\n"
    "v -2.000000 0.000000 -1.000000\n"
    "v 3.000000 0.000000 1.000000\n"
    "v 3.000000 0.000000 -1.000000\n"
    "vt 0.000000 0.000000\n"
    "vt 1.000000 0.000000\n"
    "vt 1.000000 1.000000\n"
    "vt 0.000000 1.000000\n"
    "vn 0.0000 1.0000 0.0000\n"
    "usemtl None\n"
    "s off\n"
    "f 1/1/1 2/2/1 4/3/1 3/4/1\n"
    "l 6 5\n"
    "l 4 6\n"
    "l 5 3\n"
    "l 7 8\n"
    "l 2 7\n"
    "l 8 4\n"
    "l 9 10\n"
    "l 1 9\n"
    "l 10 2\n"
    "l 12 11\n"
    "l 3 12\n"
    "l 11 1\n"
    "l 14 13\n";

TEST(cloth_remesh, MeshIO_ReadObj)
{
  MeshIO reader;
  std::istringstream stream(cube_pos_uv_normal);
  auto res = reader.read(std::move(stream), MeshIO::IOTYPE_OBJ);

  EXPECT_TRUE(res);

  const auto positions = reader.get_positions();
  const auto uvs = reader.get_uvs();
  const auto normals = reader.get_normals();
  const auto face_indices = reader.get_face_indices();
  const auto line_indices = reader.get_line_indices();

  EXPECT_EQ(positions.size(), 8);
  EXPECT_EQ(uvs.size(), 14);
  EXPECT_EQ(normals.size(), 6);
  EXPECT_EQ(face_indices.size(), 6);
  EXPECT_EQ(line_indices.size(), 0);
}

TEST(cloth_remesh, MeshIO_ReadObj_LooseEdges)
{
  MeshIO reader;
  std::istringstream stream(plane_extra_loose_edges);
  auto res = reader.read(std::move(stream), MeshIO::IOTYPE_OBJ);

  EXPECT_TRUE(res);

  const auto positions = reader.get_positions();
  const auto uvs = reader.get_uvs();
  const auto normals = reader.get_normals();
  const auto face_indices = reader.get_face_indices();
  const auto line_indices = reader.get_line_indices();

  EXPECT_EQ(positions.size(), 14);
  EXPECT_EQ(uvs.size(), 4);
  EXPECT_EQ(normals.size(), 1);
  EXPECT_EQ(face_indices.size(), 1);
  EXPECT_EQ(line_indices.size(), 13);
}

TEST(cloth_remesh, MeshIO_WriteObj)
{
  MeshIO reader;
  std::istringstream stream_in(cube_pos_uv_normal);
  auto res = reader.read(std::move(stream_in), MeshIO::IOTYPE_OBJ);
  EXPECT_TRUE(res);

  std::ostringstream stream_out;
  reader.write(stream_out, MeshIO::IOTYPE_OBJ);

  std::string expected =
      "v 1 1 -1\n"
      "v 1 -1 -1\n"
      "v 1 1 1\n"
      "v 1 -1 1\n"
      "v -1 1 -1\n"
      "v -1 -1 -1\n"
      "v -1 1 1\n"
      "v -1 -1 1\n"
      "vt 0.625 0.5\n"
      "vt 0.875 0.5\n"
      "vt 0.875 0.75\n"
      "vt 0.625 0.75\n"
      "vt 0.375 0.75\n"
      "vt 0.625 1\n"
      "vt 0.375 1\n"
      "vt 0.375 0\n"
      "vt 0.625 0\n"
      "vt 0.625 0.25\n"
      "vt 0.375 0.25\n"
      "vt 0.125 0.5\n"
      "vt 0.375 0.5\n"
      "vt 0.125 0.75\n"
      "vn 0 1 0\n"
      "vn 0 0 1\n"
      "vn -1 0 0\n"
      "vn 0 -1 0\n"
      "vn 1 0 0\n"
      "vn 0 0 -1\n"
      "f 1/1/1 5/2/1 7/3/1 3/4/1 \n"
      "f 4/5/2 3/4/2 7/6/2 8/7/2 \n"
      "f 8/8/3 7/9/3 5/10/3 6/11/3 \n"
      "f 6/12/4 2/13/4 4/5/4 8/14/4 \n"
      "f 2/13/5 1/1/5 3/4/5 4/5/5 \n"
      "f 6/11/6 5/10/6 1/1/6 2/13/6 \n";

  EXPECT_EQ(stream_out.str(), expected);
}

TEST(cloth_remesh, MeshIO_WriteObj_LooseEdges)
{
  MeshIO reader;
  std::istringstream stream_in(plane_extra_loose_edges);
  auto res = reader.read(std::move(stream_in), MeshIO::IOTYPE_OBJ);
  EXPECT_TRUE(res);

  std::ostringstream stream_out;
  reader.write(stream_out, MeshIO::IOTYPE_OBJ);

  std::string expected =
      "v -1 0 1\n"
      "v 1 0 1\n"
      "v -1 0 -1\n"
      "v 1 0 -1\n"
      "v -1 0 -2\n"
      "v 1 0 -2\n"
      "v 2 0 1\n"
      "v 2 0 -1\n"
      "v -1 0 2\n"
      "v 1 0 2\n"
      "v -2 0 1\n"
      "v -2 0 -1\n"
      "v 3 0 1\n"
      "v 3 0 -1\n"
      "vt 0 0\n"
      "vt 1 0\n"
      "vt 1 1\n"
      "vt 0 1\n"
      "vn 0 1 0\n"
      "f 1/1/1 2/2/1 4/3/1 3/4/1 \n"
      "l 6 5 \n"
      "l 4 6 \n"
      "l 5 3 \n"
      "l 7 8 \n"
      "l 2 7 \n"
      "l 8 4 \n"
      "l 9 10 \n"
      "l 1 9 \n"
      "l 10 2 \n"
      "l 12 11 \n"
      "l 3 12 \n"
      "l 11 1 \n"
      "l 14 13 \n";

  EXPECT_EQ(stream_out.str(), expected);
}

TEST(cloth_remesh, MeshIO_ReadDNAMesh)
{
  BKE_idtype_init();
  MeshIO reader;
  auto *mesh = BKE_mesh_new_nomain(0, 0, 0, 0, 0);
  CustomData_add_layer(&mesh->ldata, CD_MLOOPUV, CD_CALLOC, nullptr, 0);
  BMeshCreateParams bmesh_create_params = {0};
  bmesh_create_params.use_toolflags = true;
  BMeshFromMeshParams bmesh_from_mesh_params = {0};
  bmesh_from_mesh_params.cd_mask_extra = CD_MASK_MESH;

  auto *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  BMO_op_callf(bm, (BMO_FLAG_DEFAULTS & ~BMO_FLAG_RESPECT_HIDE), "create_cube calc_uvs=%b", true);

  BMeshToMeshParams bmesh_to_mesh_params = {0};
  bmesh_to_mesh_params.cd_mask_extra = CD_MASK_MESH;

  auto *bm_copy = BM_mesh_copy(bm);

  auto *result = BKE_mesh_new_nomain(0, 0, 0, 0, 0);
  BM_mesh_bm_to_me(nullptr, bm_copy, result, &bmesh_to_mesh_params);

  BKE_mesh_calc_normals(result);

  auto res = reader.read(result);

  EXPECT_TRUE(res);

  BM_mesh_free(bm);
  BM_mesh_free(bm_copy);
  BKE_mesh_eval_delete(mesh);
  BKE_mesh_eval_delete(result);

  std::ostringstream stream_out;
  reader.write(stream_out, MeshIO::IOTYPE_OBJ);

  std::string expected =
      "v -0.5 -0.5 -0.5\n"
      "v -0.5 -0.5 0.5\n"
      "v -0.5 0.5 -0.5\n"
      "v -0.5 0.5 0.5\n"
      "v 0.5 -0.5 -0.5\n"
      "v 0.5 -0.5 0.5\n"
      "v 0.5 0.5 -0.5\n"
      "v 0.5 0.5 0.5\n"
      "vt 0.375 0\n"
      "vt 0.625 0\n"
      "vt 0.625 0.25\n"
      "vt 0.375 0.25\n"
      "vt 0.375 0.25\n"
      "vt 0.625 0.25\n"
      "vt 0.625 0.5\n"
      "vt 0.375 0.5\n"
      "vt 0.375 0.5\n"
      "vt 0.625 0.5\n"
      "vt 0.625 0.75\n"
      "vt 0.375 0.75\n"
      "vt 0.375 0.75\n"
      "vt 0.625 0.75\n"
      "vt 0.625 1\n"
      "vt 0.375 1\n"
      "vt 0.125 0.5\n"
      "vt 0.375 0.5\n"
      "vt 0.375 0.75\n"
      "vt 0.125 0.75\n"
      "vt 0.625 0.5\n"
      "vt 0.875 0.5\n"
      "vt 0.875 0.75\n"
      "vt 0.625 0.75\n"
      "vn -0.577349 -0.577349 -0.577349\n"
      "vn -0.577349 -0.577349 0.577349\n"
      "vn -0.577349 0.577349 -0.577349\n"
      "vn -0.577349 0.577349 0.577349\n"
      "vn 0.577349 -0.577349 -0.577349\n"
      "vn 0.577349 -0.577349 0.577349\n"
      "vn 0.577349 0.577349 -0.577349\n"
      "vn 0.577349 0.577349 0.577349\n"
      "f 1/1/1 2/2/2 4/3/4 3/4/3 \n"
      "f 3/5/3 4/6/4 8/7/8 7/8/7 \n"
      "f 7/9/7 8/10/8 6/11/6 5/12/5 \n"
      "f 5/13/5 6/14/6 2/15/2 1/16/1 \n"
      "f 3/17/3 7/18/7 5/19/5 1/20/1 \n"
      "f 8/21/8 4/22/4 2/23/2 6/24/6 \n";

  EXPECT_EQ(stream_out.str(), expected);
}

TEST(cloth_remesh, MeshIO_WriteDNAMesh)
{
  MeshIO reader;
  std::istringstream stream_in(cube_pos_uv_normal);
  auto res = reader.read(std::move(stream_in), MeshIO::IOTYPE_OBJ);
  EXPECT_TRUE(res);

  auto *mesh = reader.write();
  EXPECT_NE(mesh, nullptr);

  EXPECT_NE(mesh->mvert, nullptr);
  EXPECT_NE(mesh->medge, nullptr);
  EXPECT_NE(mesh->mpoly, nullptr);
  EXPECT_NE(mesh->mloop, nullptr);
  EXPECT_NE(mesh->mloopuv, nullptr);

  auto format_string_mvert = [](const MVert &mvert) {
    std::ostringstream stream;
    stream << "[mvert: ("
           << "co: " << stringify_v3(mvert.co) << ")]";
    return stream.str();
  };
  auto format_string_mloopuv = [](const MLoopUV &mloopuv) {
    std::ostringstream stream;
    stream << "[mloopuv: ("
           << "uv: " << stringify_v2(mloopuv.uv) << ")]";
    return stream.str();
  };
  auto format_string_medge = [](const MEdge &medge) {
    std::ostringstream stream;
    stream << "[medge: ("
           << "v1: " << medge.v1 << ", v2: " << medge.v2 << ")]";
    return stream.str();
  };
  auto format_string_mpoly = [](const MPoly &mpoly) {
    std::ostringstream stream;
    stream << "[mpoly: ("
           << "loopstart: " << mpoly.loopstart << ", totloop: " << mpoly.totloop << ")]";
    return stream.str();
  };
  auto format_string_mloop = [](const MLoop &mloop) {
    std::ostringstream stream;
    stream << "[mloop: ("
           << "v: " << mloop.v << ", e: " << mloop.e << ")]";
    return stream.str();
  };

  std::string expected =
      "[mvert: (co: (1, 1, -1))]\n"
      "[mvert: (co: (1, -1, -1))]\n"
      "[mvert: (co: (1, 1, 1))]\n"
      "[mvert: (co: (1, -1, 1))]\n"
      "[mvert: (co: (-1, 1, -1))]\n"
      "[mvert: (co: (-1, -1, -1))]\n"
      "[mvert: (co: (-1, 1, 1))]\n"
      "[mvert: (co: (-1, -1, 1))]\n"
      "[mloopuv: (uv: (0.625, 0.5))]\n"
      "[mloopuv: (uv: (0.875, 0.5))]\n"
      "[mloopuv: (uv: (0.875, 0.75))]\n"
      "[mloopuv: (uv: (0.625, 0.75))]\n"
      "[mloopuv: (uv: (0.375, 0.75))]\n"
      "[mloopuv: (uv: (0.625, 0.75))]\n"
      "[mloopuv: (uv: (0.625, 1))]\n"
      "[mloopuv: (uv: (0.375, 1))]\n"
      "[mloopuv: (uv: (0.375, 0))]\n"
      "[mloopuv: (uv: (0.625, 0))]\n"
      "[mloopuv: (uv: (0.625, 0.25))]\n"
      "[mloopuv: (uv: (0.375, 0.25))]\n"
      "[mloopuv: (uv: (0.125, 0.5))]\n"
      "[mloopuv: (uv: (0.375, 0.5))]\n"
      "[mloopuv: (uv: (0.375, 0.75))]\n"
      "[mloopuv: (uv: (0.125, 0.75))]\n"
      "[mloopuv: (uv: (0.375, 0.5))]\n"
      "[mloopuv: (uv: (0.625, 0.5))]\n"
      "[mloopuv: (uv: (0.625, 0.75))]\n"
      "[mloopuv: (uv: (0.375, 0.75))]\n"
      "[mloopuv: (uv: (0.375, 0.25))]\n"
      "[mloopuv: (uv: (0.625, 0.25))]\n"
      "[mloopuv: (uv: (0.625, 0.5))]\n"
      "[mloopuv: (uv: (0.375, 0.5))]\n"
      "[mpoly: (loopstart: 0, totloop: 4)]\n"
      "[mpoly: (loopstart: 4, totloop: 4)]\n"
      "[mpoly: (loopstart: 8, totloop: 4)]\n"
      "[mpoly: (loopstart: 12, totloop: 4)]\n"
      "[mpoly: (loopstart: 16, totloop: 4)]\n"
      "[mpoly: (loopstart: 20, totloop: 4)]\n"
      "[mloop: (v: 0, e: 3)]\n"
      "[mloop: (v: 4, e: 5)]\n"
      "[mloop: (v: 6, e: 9)]\n"
      "[mloop: (v: 2, e: 1)]\n"
      "[mloop: (v: 3, e: 2)]\n"
      "[mloop: (v: 2, e: 9)]\n"
      "[mloop: (v: 6, e: 10)]\n"
      "[mloop: (v: 7, e: 6)]\n"
      "[mloop: (v: 7, e: 10)]\n"
      "[mloop: (v: 6, e: 5)]\n"
      "[mloop: (v: 4, e: 4)]\n"
      "[mloop: (v: 5, e: 8)]\n"
      "[mloop: (v: 5, e: 7)]\n"
      "[mloop: (v: 1, e: 11)]\n"
      "[mloop: (v: 3, e: 6)]\n"
      "[mloop: (v: 7, e: 8)]\n"
      "[mloop: (v: 1, e: 0)]\n"
      "[mloop: (v: 0, e: 1)]\n"
      "[mloop: (v: 2, e: 2)]\n"
      "[mloop: (v: 3, e: 11)]\n"
      "[mloop: (v: 5, e: 4)]\n"
      "[mloop: (v: 4, e: 3)]\n"
      "[mloop: (v: 0, e: 0)]\n"
      "[mloop: (v: 1, e: 7)]\n"
      "[medge: (v1: 0, v2: 1)]\n"
      "[medge: (v1: 0, v2: 2)]\n"
      "[medge: (v1: 2, v2: 3)]\n"
      "[medge: (v1: 0, v2: 4)]\n"
      "[medge: (v1: 4, v2: 5)]\n"
      "[medge: (v1: 4, v2: 6)]\n"
      "[medge: (v1: 3, v2: 7)]\n"
      "[medge: (v1: 1, v2: 5)]\n"
      "[medge: (v1: 5, v2: 7)]\n"
      "[medge: (v1: 2, v2: 6)]\n"
      "[medge: (v1: 6, v2: 7)]\n"
      "[medge: (v1: 1, v2: 3)]\n";

  std::ostringstream sout;
  for (auto i = 0; i < mesh->totvert; i++) {
    sout << format_string_mvert(mesh->mvert[i]) << std::endl;
  }
  for (auto i = 0; i < mesh->totloop; i++) {
    sout << format_string_mloopuv(mesh->mloopuv[i]) << std::endl;
  }
  for (auto i = 0; i < mesh->totpoly; i++) {
    sout << format_string_mpoly(mesh->mpoly[i]) << std::endl;
  }
  for (auto i = 0; i < mesh->totloop; i++) {
    sout << format_string_mloop(mesh->mloop[i]) << std::endl;
  }
  for (auto i = 0; i < mesh->totedge; i++) {
    sout << format_string_medge(mesh->medge[i]) << std::endl;
  }

  EXPECT_EQ(sout.str(), expected);

  BKE_mesh_eval_delete(mesh);
}

TEST(cloth_remesh, Mesh_Read)
{
  MeshIO reader;
  std::istringstream stream(cube_pos_uv_normal);
  reader.read(std::move(stream), MeshIO::IOTYPE_OBJ);

  Mesh<bool, bool, bool, bool> mesh;
  mesh.read(reader);

  const auto &nodes = mesh.get_nodes();
  const auto &verts = mesh.get_verts();
  const auto &edges = mesh.get_edges();
  const auto &faces = mesh.get_faces();

  EXPECT_EQ(nodes.size(), 8);
  EXPECT_EQ(verts.size(), 14);
  EXPECT_EQ(edges.size(), 19);
  EXPECT_EQ(faces.size(), 6);

  for (const auto &face : faces) {
    EXPECT_EQ(face.get_verts().size(), 4);
  }

  for (const auto &edge : edges) {
    auto num_faces = edge.get_faces().size();
    EXPECT_TRUE(num_faces == 1 || num_faces == 2);
  }

  for (const auto &vert : verts) {
    auto num_edges = vert.get_edges().size();
    EXPECT_TRUE(num_edges >= 2 && num_edges <= 4);
    EXPECT_NE(vert.get_node(), std::nullopt);
  }

  for (const auto &node : nodes) {
    auto num_verts = node.get_verts().size();
    EXPECT_TRUE(num_verts >= 1 && num_verts <= 3);
  }
}

TEST(cloth_remesh, Mesh_Write)
{
  MeshIO reader;
  std::istringstream stream(cube_pos_uv_normal);
  reader.read(std::move(stream), MeshIO::IOTYPE_OBJ);

  Mesh<bool, bool, bool, bool> mesh;
  mesh.read(reader);

  auto result = mesh.write();

  const auto positions = result.get_positions();
  const auto uvs = result.get_uvs();
  const auto normals = result.get_normals();
  const auto face_indices = result.get_face_indices();
  const auto line_indices = result.get_line_indices();

  /* TODO(ish): add some more complex checks, it should fail when the
   * `Mesh` had gaps in the arena, say after some collapse edge operation */

  EXPECT_EQ(positions.size(), 8);
  EXPECT_EQ(uvs.size(), 14);
  EXPECT_EQ(normals.size(), 8);
  EXPECT_EQ(face_indices.size(), 6);
  EXPECT_EQ(line_indices.size(), 0);
}

} /* namespace blender::bke::tests */
