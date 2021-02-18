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
 *
 * The Original Code is Copyright (C) 2020 Blender Foundation.
 * All rights reserved.
 */

#include "usd_material_importer.h"

#include "DNA_material_types.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "BKE_image.h"
#include "BKE_lib_id.h"
#include "BKE_main.h"
#include "BKE_material.h"
#include "BKE_mesh.h"
#include "BKE_node.h"
#include "BKE_object.h"

#include "BLI_listbase.h"
#include "BLI_math_vector.h"
#include "BLI_string.h"

#include <pxr/base/gf/vec3f.h>

#include <iostream>
#include <vector>

namespace usdtokens {

// Parameter names
static const pxr::TfToken a("a", pxr::TfToken::Immortal);
static const pxr::TfToken b("b", pxr::TfToken::Immortal);
static const pxr::TfToken clearcoat("clearcoat", pxr::TfToken::Immortal);
static const pxr::TfToken clearcoatRoughness("clearcoatRoughness", pxr::TfToken::Immortal);
static const pxr::TfToken diffuseColor("diffuseColor", pxr::TfToken::Immortal);
static const pxr::TfToken emissiveColor("emissiveColor", pxr::TfToken::Immortal);
static const pxr::TfToken file("file", pxr::TfToken::Immortal);
static const pxr::TfToken g("g", pxr::TfToken::Immortal);
static const pxr::TfToken ior("ior", pxr::TfToken::Immortal);
static const pxr::TfToken metallic("metallic", pxr::TfToken::Immortal);
static const pxr::TfToken normal("normal", pxr::TfToken::Immortal);
static const pxr::TfToken occlusion("occlusion", pxr::TfToken::Immortal);
static const pxr::TfToken opacity("opacity", pxr::TfToken::Immortal);
static const pxr::TfToken r("r", pxr::TfToken::Immortal);
static const pxr::TfToken result("result", pxr::TfToken::Immortal);
static const pxr::TfToken rgb("rgb", pxr::TfToken::Immortal);
static const pxr::TfToken rgba("rgba", pxr::TfToken::Immortal);
static const pxr::TfToken roughness("roughness", pxr::TfToken::Immortal);
static const pxr::TfToken specularColor("specularColor", pxr::TfToken::Immortal);
static const pxr::TfToken st("st", pxr::TfToken::Immortal);
static const pxr::TfToken varname("varname", pxr::TfToken::Immortal);

// Color space names
static const pxr::TfToken RAW("RAW", pxr::TfToken::Immortal);

// USD shader names.
static const pxr::TfToken UsdPreviewSurface("UsdPreviewSurface", pxr::TfToken::Immortal);
static const pxr::TfToken UsdPrimvarReader_float2("UsdPrimvarReader_float2",
                                                  pxr::TfToken::Immortal);
static const pxr::TfToken UsdUVTexture("UsdUVTexture", pxr::TfToken::Immortal);
}  // namespace usdtokens

static bNode *add_node(const bContext *C, bNodeTree *ntree, int type, float locx, float locy)
{
  bNode *new_node = nodeAddStaticNode(C, ntree, type);

  if (new_node) {
    new_node->locx = locx;
    new_node->locy = locy;
  }

  return new_node;
}

static void link_nodes(
    bNodeTree *ntree, bNode *source, const char *sock_out, bNode *dest, const char *sock_in)
{
  bNodeSocket *source_socket = nodeFindSocket(source, SOCK_OUT, sock_out);

  if (!source_socket) {
    std::cerr << "PROGRAMMER ERROR: Couldn't find output socket " << sock_out << std::endl;
    return;
  }

  bNodeSocket *dest_socket = nodeFindSocket(dest, SOCK_IN, sock_in);

  if (!dest_socket) {
    std::cerr << "PROGRAMMER ERROR: Couldn't find input socket " << sock_in << std::endl;
    return;
  }

  nodeAddLink(ntree, source, source_socket, dest, dest_socket);
}

static pxr::UsdShadeShader get_source_shader(const pxr::UsdShadeConnectableAPI &source,
                                             pxr::TfToken in_shader_id)
{
  if (source && source.IsShader()) {
    pxr::UsdShadeShader source_shader(source.GetPrim());
    if (source_shader) {
      pxr::TfToken shader_id;
      if (source_shader.GetShaderId(&shader_id) && shader_id == in_shader_id) {
        return source_shader;
      }
    }
  }
  return pxr::UsdShadeShader();
}

namespace blender::io::usd {

namespace {

// Compute the x- and y-coordinates for placing a new node in an unoccupied region of
// the column with the given index.  Returns the coordinates in r_locx and r_locy and
// updates the column-occupancy information in r_ctx.
void compute_node_loc(
    int column, float node_height, float &r_locx, float &r_locy, NodePlacementContext &r_ctx)
{
  r_locx = r_ctx.origx - column * r_ctx.horizontal_step;

  if (column >= r_ctx.column_offsets.size()) {
    r_ctx.column_offsets.push_back(0.0f);
  }

  r_locy = r_ctx.origy - r_ctx.column_offsets[column];

  // Record the y-offset of the occupied region in
  // the column, including padding.
  r_ctx.column_offsets[column] += node_height + 10.0f;
}

}  // namespace

USDMaterialImporter::USDMaterialImporter(const USDImporterContext &context, Main *bmain)
    : context_(context), bmain_(bmain)
{
}

Material *USDMaterialImporter::add_material(const pxr::UsdShadeMaterial &usd_material) const
{
  if (!(bmain_ && usd_material)) {
    return nullptr;
  }

  std::string mtl_name = usd_material.GetPrim().GetName().GetString().c_str();

  /* Create the material. */
  Material *mtl = BKE_material_add(bmain_, mtl_name.c_str());

  /* Optionally, create shader nodes to represent a UsdPreviewSurface. */
  if (context_.import_params.import_usdpreview) {
    import_usd_preview(mtl, usd_material);
  }

  return mtl;
}

/* Convert a UsdPreviewSurface shader network to Blender nodes.
 * The logic doesn't yet handle converting arbitrary prim var reader nodes. */

void USDMaterialImporter::import_usd_preview(Material *mtl,
                                             const pxr::UsdShadeMaterial &usd_material) const
{
  if (!usd_material) {
    return;
  }

  /* Get the surface shader. */
  pxr::UsdShadeShader surf_shader = usd_material.ComputeSurfaceSource();

  if (surf_shader) {
    /* Check if we have a UsdPreviewSurface shader. */
    pxr::TfToken shader_id;
    if (surf_shader.GetShaderId(&shader_id) && shader_id == usdtokens::UsdPreviewSurface) {
      import_usd_preview(mtl, surf_shader);
    }
  }
}

/* Create the Principled BSDF shader node network. */
void USDMaterialImporter::import_usd_preview(Material *mtl,
                                             const pxr::UsdShadeShader &usd_shader) const
{
  if (!(bmain_ && mtl && usd_shader)) {
    return;
  }

  /* Create the Material's node tree containing the principled
   * and output shader. */

  bNodeTree *ntree = ntreeAddTree(NULL, "Shader Nodetree", "ShaderNodeTree");
  mtl->nodetree = ntree;
  mtl->use_nodes = true;

  bNode *principled = add_node(NULL, ntree, SH_NODE_BSDF_PRINCIPLED, 0.0f, 300.0f);

  if (!principled) {
    std::cerr << "ERROR: Couldn't create SH_NODE_BSDF_PRINCIPLED node for USD shader "
              << usd_shader.GetPath() << std::endl;
    return;
  }

  bNode *output = add_node(NULL, ntree, SH_NODE_OUTPUT_MATERIAL, 300.0f, 300.0f);

  if (!output) {
    std::cerr << "ERROR: Couldn't create SH_NODE_OUTPUT_MATERIAL node for USD shader "
              << usd_shader.GetPath() << std::endl;
    return;
  }

  link_nodes(ntree, principled, "BSDF", output, "Surface");

  /* Set up the principled shader inputs. */

  /* The following keep track of the locations for adding
   * input nodes. */

  NodePlacementContext context(0.0f, 300.0);
  int column = 0;

  /* Set the principled shader inputs. */

  if (pxr::UsdShadeInput diffuse_input = usd_shader.GetInput(usdtokens::diffuseColor)) {
    set_node_input(diffuse_input, principled, "Base Color", ntree, column, context);
  }

  if (pxr::UsdShadeInput emissive_input = usd_shader.GetInput(usdtokens::emissiveColor)) {
    set_node_input(emissive_input, principled, "Emission", ntree, column, context);
  }

  if (pxr::UsdShadeInput specular_input = usd_shader.GetInput(usdtokens::specularColor)) {
    set_node_input(specular_input, principled, "Specular", ntree, column, context);
  }

  if (pxr::UsdShadeInput metallic_input = usd_shader.GetInput(usdtokens::metallic)) {
    ;
    set_node_input(metallic_input, principled, "Metallic", ntree, column, context);
  }

  if (pxr::UsdShadeInput roughness_input = usd_shader.GetInput(usdtokens::roughness)) {
    set_node_input(roughness_input, principled, "Roughness", ntree, column, context);
  }

  if (pxr::UsdShadeInput clearcoat_input = usd_shader.GetInput(usdtokens::clearcoat)) {
    set_node_input(clearcoat_input, principled, "Clearcoat", ntree, column, context);
  }

  if (pxr::UsdShadeInput clearcoat_roughness_input = usd_shader.GetInput(
          usdtokens::clearcoatRoughness)) {
    set_node_input(
        clearcoat_roughness_input, principled, "Clearcoat Roughness", ntree, column, context);
  }

  if (pxr::UsdShadeInput opacity_input = usd_shader.GetInput(usdtokens::opacity)) {
    set_node_input(opacity_input, principled, "Alpha", ntree, column, context);
  }

  if (pxr::UsdShadeInput ior_input = usd_shader.GetInput(usdtokens::ior)) {
    set_node_input(ior_input, principled, "IOR", ntree, column, context);
  }

  if (pxr::UsdShadeInput normal_input = usd_shader.GetInput(usdtokens::normal)) {
    set_node_input(normal_input, principled, "Normal", ntree, column, context);
  }

  nodeSetActive(ntree, output);
}

/* Convert the given USD shader input to an input on the given node. */
void USDMaterialImporter::set_node_input(const pxr::UsdShadeInput &usd_input,
                                         bNode *dest_node,
                                         const char *dest_socket_name,
                                         bNodeTree *ntree,
                                         int column,
                                         NodePlacementContext &r_ctx) const
{
  if (!(usd_input && dest_node)) {
    return;
  }

  if (usd_input.HasConnectedSource()) {
    pxr::UsdShadeConnectableAPI source;
    pxr::TfToken source_name;
    pxr::UsdShadeAttributeType source_type;

    usd_input.GetConnectedSource(&source, &source_name, &source_type);

    if (!(source && source.IsShader())) {
      return;
    }

    pxr::UsdShadeShader source_shader(source.GetPrim());

    if (!source_shader) {
      return;
    }

    pxr::TfToken shader_id;
    if (!source_shader.GetShaderId(&shader_id)) {
      std::cerr << "ERROR: couldn't get shader id for source shader "
                << source_shader.GetPrim().GetPath() << std::endl;
      return;
    }

    /* For now, only convert UsdUVTexture and UsdPrimvarReader_float2 inputs. */
    if (shader_id == usdtokens::UsdUVTexture) {

      if (strcmp(dest_socket_name, "Normal") == 0) {

        // The normal texture input requires creating a normal map node.
        float locx = 0.0f;
        float locy = 0.0f;
        compute_node_loc(column + 1, 300.0, locx, locy, r_ctx);

        bNode *normal_map = add_node(NULL, ntree, SH_NODE_NORMAL_MAP, locx, locy);

        // Currently, the Normal Map node has Tangent Space as the default,
        // which is what we need, so we don't need to explicitly set it.

        // Connect the Normal Map to the Normal input.
        link_nodes(ntree, normal_map, "Normal", dest_node, "Normal");

        // Now, create the Texture Image node input to the Normal Map "Color" input.
        convert_usd_uv_texture(
            source_shader, source_name, normal_map, "Color", ntree, column + 2, r_ctx);
      }
      else {
        convert_usd_uv_texture(
            source_shader, source_name, dest_node, dest_socket_name, ntree, column + 1, r_ctx);
      }
    }
    else if (shader_id == usdtokens::UsdPrimvarReader_float2) {
      convert_usd_primvar_reader(
          source_shader, source_name, dest_node, dest_socket_name, ntree, column + 1, r_ctx);
    }
  }
  else {
    bNodeSocket *sock = nodeFindSocket(dest_node, SOCK_IN, dest_socket_name);
    if (!sock) {
      std::cerr << "ERROR: couldn't get destination node socket " << dest_socket_name << std::endl;
      return;
    }

    pxr::VtValue val;
    if (!usd_input.Get(&val)) {
      std::cerr << "ERROR: couldn't get value for usd shader input "
                << usd_input.GetPrim().GetPath() << std::endl;
      return;
    }

    switch (sock->type) {
      case SOCK_FLOAT:
        if (val.IsHolding<float>()) {
          ((bNodeSocketValueFloat *)sock->default_value)->value = val.UncheckedGet<float>();
        }
        else if (val.IsHolding<pxr::GfVec3f>()) {
          pxr::GfVec3f v3f = val.UncheckedGet<pxr::GfVec3f>();
          float average = (v3f[0] + v3f[1] + v3f[2]) / 3.0f;
          ((bNodeSocketValueFloat *)sock->default_value)->value = average;
        }
        break;
      case SOCK_RGBA:
        if (val.IsHolding<pxr::GfVec3f>()) {
          pxr::GfVec3f v3f = val.UncheckedGet<pxr::GfVec3f>();
          copy_v3_v3(((bNodeSocketValueRGBA *)sock->default_value)->value, v3f.data());
        }
        break;
      case SOCK_VECTOR:
        if (val.IsHolding<pxr::GfVec3f>()) {
          pxr::GfVec3f v3f = val.UncheckedGet<pxr::GfVec3f>();
          copy_v3_v3(((bNodeSocketValueVector *)sock->default_value)->value, v3f.data());
        }
        else if (val.IsHolding<pxr::GfVec2f>()) {
          pxr::GfVec2f v2f = val.UncheckedGet<pxr::GfVec2f>();
          copy_v2_v2(((bNodeSocketValueVector *)sock->default_value)->value, v2f.data());
        }
        break;
      default:
        std::cerr << "WARNING: unexpected type " << sock->idname << " for destination node socket "
                  << dest_socket_name << std::endl;
        break;
    }
  }
}

void USDMaterialImporter::convert_usd_uv_texture(const pxr::UsdShadeShader &usd_shader,
                                                 const pxr::TfToken &usd_source_name,
                                                 bNode *dest_node,
                                                 const char *dest_socket_name,
                                                 bNodeTree *ntree,
                                                 int column,
                                                 NodePlacementContext &r_ctx) const
{
  if (!usd_shader || !dest_node || !ntree || !dest_socket_name || !bmain_) {
    return;
  }

  float locx = 0.0f;
  float locy = 0.0f;
  compute_node_loc(column, 300.0, locx, locy, r_ctx);

  // Create the Texture Image node.
  bNode *tex_image = add_node(NULL, ntree, SH_NODE_TEX_IMAGE, locx, locy);

  if (!tex_image) {
    std::cerr << "ERROR: Couldn't create SH_NODE_TEX_IMAGE for node input " << dest_socket_name
              << std::endl;
    return;
  }

  // Try to load the texture image.
  pxr::UsdShadeInput file_input = usd_shader.GetInput(usdtokens::file);
  if (file_input) {

    pxr::VtValue file_val;
    if (file_input.Get(&file_val) && file_val.IsHolding<pxr::SdfAssetPath>()) {
      const pxr::SdfAssetPath &asset_path = file_val.Get<pxr::SdfAssetPath>();
      std::string file_path = asset_path.GetResolvedPath();
      if (!file_path.empty()) {
        const char *im_file = file_path.c_str();
        Image *image = BKE_image_load_exists(bmain_, im_file);
        if (image) {
          tex_image->id = &image->id;

          // Set texture color space.
          // TODO(makowalski): For now, just checking for RAW color space,
          // assuming sRGB otherwise, but more complex logic might be
          // required if the color space is "auto".
          pxr::TfToken colorSpace = file_input.GetAttr().GetColorSpace();
          if (colorSpace == usdtokens::RAW) {
            STRNCPY(image->colorspace_settings.name, "Raw");
          }
        }
        else {
          std::cerr << "WARNING: Couldn't open image file '" << im_file
                    << "' for Texture Image node." << std::endl;
        }
      }
      else {
        std::cerr << "WARNING: Couldn't resolve image asset '" << asset_path
                  << "' for Texture Image node." << std::endl;
      }
    }
  }

  // Connect to destination node input.

  // Get the source socket name.
  std::string source_socket_name = usd_source_name == usdtokens::a ? "Alpha" : "Color";

  link_nodes(ntree, tex_image, source_socket_name.c_str(), dest_node, dest_socket_name);

  // Connect the texture image node "Vector" input.
  if (pxr::UsdShadeInput st_input = usd_shader.GetInput(usdtokens::st)) {
    set_node_input(st_input, tex_image, "Vector", ntree, column, r_ctx);
  }
}

void USDMaterialImporter::convert_usd_primvar_reader(const pxr::UsdShadeShader &usd_shader,
                                                     const pxr::TfToken &usd_source_name,
                                                     bNode *dest_node,
                                                     const char *dest_socket_name,
                                                     bNodeTree *ntree,
                                                     int column,
                                                     NodePlacementContext &r_ctx) const
{
  if (!usd_shader || !dest_node || !ntree || !dest_socket_name || !bmain_) {
    return;
  }

  float locx = 0.0f;
  float locy = 0.0f;
  compute_node_loc(column, 300.0f, locx, locy, r_ctx);

  // Create the UV Map node.
  bNode *uv_map = add_node(NULL, ntree, SH_NODE_UVMAP, locx, locy);

  if (!uv_map) {
    std::cerr << "ERROR: Couldn't create SH_NODE_UVMAP for node input " << dest_socket_name
              << std::endl;
    return;
  }

  // Set the texmap name.
  pxr::UsdShadeInput varname_input = usd_shader.GetInput(usdtokens::varname);
  if (varname_input) {
    pxr::VtValue varname_val;
    if (varname_input.Get(&varname_val) && varname_val.IsHolding<pxr::TfToken>()) {
      std::string varname = varname_val.Get<pxr::TfToken>().GetString();
      if (!varname.empty()) {
        NodeShaderUVMap *storage = (NodeShaderUVMap *)uv_map->storage;
        BLI_strncpy(storage->uv_map, varname.c_str(), sizeof(storage->uv_map));
      }
    }
  }

  // Connect to destination node input.
  link_nodes(ntree, uv_map, "UV", dest_node, dest_socket_name);
}

}  // namespace blender::io::usd
