float cross_tri_v2(vec2 v1, vec2 v2, vec2 v3)
{
  return (v1.x - v2.x) * (v2.y - v3.y) + (v1.y - v2.y) * (v3.x - v2.x);
}

void main() {
    ivec2 pixel_coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(pixels);

    vec2 pixel_uv = vec2(pixel_coord) / vec2(image_size);
    int pixel_index = pixel_coord.y * image_size.x + pixel_coord.x;

    TexturePaintPixel pixel;
    pixel.pbvh_node_index = -1;
    pixel.poly_index = -1;

    for (int poly_index = 0; poly_index < num_polygons; poly_index ++) 
    {
        TexturePaintPolygon poly = polygons[poly_index];
        vec3 weights = vec3(
            cross_tri_v2(poly.uv[1], poly.uv[2], pixel_uv),
            cross_tri_v2(poly.uv[2], poly.uv[0], pixel_uv),
            cross_tri_v2(poly.uv[0], poly.uv[1], pixel_uv)
        );

        float tot_weight = weights.x + weights.y + weights.z;
        if (tot_weight == 0.0f) {
            continue;
        }

        weights /= tot_weight;
        if (any(lessThan(weights, vec3(0.0)))|| any(greaterThan(weights, vec3(1.0)))) {
            /* No barycentric intersection detected with current polygon, continue with next.*/
            continue;
        }

        /* Found solution for this pixel. */
        pixel.pbvh_node_index = poly.pbvh_node_index;
        pixel.poly_index = poly_index;
        break;
    }

  imageStore(pixels, pixel_coord, ivec4(pixel.pbvh_node_index, pixel.poly_index, 0, 0));
}