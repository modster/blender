
struct MeshData {
  /** World position. */
  vec3 P;
  /** Surface Normal. */
  vec3 N;
  /** Geometric Normal. */
  vec3 Ng;
  /** Barycentric coordinates. */
  vec2 barycentrics;
};

IN_OUT MeshDataInterface
{
  vec3 P;
  vec3 N;
}
interp;
