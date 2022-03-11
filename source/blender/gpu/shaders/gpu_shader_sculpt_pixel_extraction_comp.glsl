float cross_tri_v2(vec2 v1, vec2 v2, vec2 v3)
{
  return (v1.x - v2.x) * (v2.y - v3.y) + (v1.y - v2.y) * (v3.x - v2.x);
}

void main()
{
  ivec2 pixel_coord = ivec2(gl_GlobalInvocationID.xy);
  ivec2 image_size = imageSize(pixels);

  vec2 pixel_uv = vec2(pixel_coord) / vec2(image_size);

  if (imageLoad(pixels, pixel_coord).x != -1) {
    return;
  };

  for (int poly_index = from_polygon; poly_index < to_polygon; poly_index++) {
    vec2 uv0 = polygons[poly_index * 2].xy;
    vec2 uv1 = polygons[poly_index * 2].zw;
    vec2 uv2 = polygons[poly_index * 2 + 1].xy;
    int pbvh_node_index = floatBitsToInt(polygons[poly_index * 2 + 1].z);
    int pbvh_poly_index = floatBitsToInt(polygons[poly_index * 2 + 1].w);

    vec3 weights = vec3(cross_tri_v2(uv1, uv2, pixel_uv),
                        cross_tri_v2(uv2, uv0, pixel_uv),
                        cross_tri_v2(uv0, uv1, pixel_uv));

    float tot_weight = weights.x + weights.y + weights.z;
    if (tot_weight == 0.0) {
      continue;
    }

    weights /= tot_weight;
    if (any(lessThan(weights, vec3(0.0))) || any(greaterThan(weights, vec3(1.0)))) {
      /* No barycentric intersection detected with current polygon, continue with next.*/
      continue;
    }

    /* Found solution for this pixel. */
    ivec4 pixel = ivec4(pbvh_node_index, pbvh_poly_index, 0, 255);
    imageStore(pixels, pixel_coord, pixel);
    break;
  }
}