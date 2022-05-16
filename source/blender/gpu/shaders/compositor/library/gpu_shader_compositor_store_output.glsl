/* The GLSL specification is not clear about passing images to functions and consequently functions
 * with image parameters are not portable across driver implementations or even non-functioning in
 * some drivers. See https://github.com/KhronosGroup/GLSL/issues/57.
 * To work around this, we use macros instead of functions. However, to make those macros usable in
 * the GPU material library, we also define function counterparts that are guarded with #if 0 such
 * that they are not used in the shader but are parsed by the GPU shader dependency parser. */

#if 0
void store_output_float(restrict writeonly image2D output_image, float value)
{
  imageStore(output_image, ivec2(gl_GlobalInvocationID.xy), vec4(value));
}

void store_output_vector(restrict writeonly image2D output_image, vec3 vector)
{
  imageStore(output_image, ivec2(gl_GlobalInvocationID.xy), vec4(vector, 0.0));
}

void store_output_color(restrict writeonly image2D output_image, vec4 color)
{
  imageStore(output_image, ivec2(gl_GlobalInvocationID.xy), color);
}
#else
#  define store_output_float(output_image, value) \
    imageStore(output_image, ivec2(gl_GlobalInvocationID.xy), vec4(value));

#  define store_output_vector(output_image, vector) \
    imageStore(output_image, ivec2(gl_GlobalInvocationID.xy), vec4(vector, 0.0));

#  define store_output_color(output_image, color) \
    imageStore(output_image, ivec2(gl_GlobalInvocationID.xy), color);
#endif
