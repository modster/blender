

/* ---------------------------------------------------------------------- */
/** \name Shadow Sampling Functions
 * \{ */

/* Turns local light coordinate into shadow region index. Matches eShadowCubeFace order. */
int shadow_punctual_region_get(vec3 lL)
{
  vec3 aP = abs(lL);
  if (all(greaterThan(aP.xx, aP.yz))) {
    return (lL.x > 0.0) ? 1 : 2;
  }
  else if (all(greaterThan(aP.yy, aP.xz))) {
    return (lL.y > 0.0) ? 3 : 4;
  }
  else {
    return (lL.z > 0.0) ? 5 : 0;
  }
}

/** \} */
