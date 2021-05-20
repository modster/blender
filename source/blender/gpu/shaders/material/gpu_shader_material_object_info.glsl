void node_object_info(float mat_index,
                      out vec3 location,
                      out vec4 color,
                      out float object_index,
                      out float material_index,
                      out float random)
{
  location = ModelMatrix[3].xyz;
  color = ObjectColor;
  object_index = ObjectInfo.x;
  material_index = mat_index;
  random = ObjectInfo.z;
}
