
/**
 * Debug print implementation for shaders.
 *
 * `print()`:
 *   Log variable or strings inside the viewport.
 *   Using a unique non string argument will print the variable name with it.
 *   Concatenate by using multiple arguments. i.e: `print("Looped ", n, " times.")`.
 * `print_no_endl()`:
 *   Same as `print()` but does not finish the line.
 * `print_value()`:
 *   Display only the value of a variable. Does not finish the line.
 * `print_value_hex()`:
 *   Display only the hex representation of a variable. Does not finish the line.
 * `print_value_binary()`:
 *   Display only the binary representation of a variable. Does not finish the line.
 *
 * IMPORTANT: As it is now, it is not yet thread safe. Only print from one thread. You can use the
 * IS_DEBUG_MOUSE_FRAGMENT macro in fragment shader to filter using mouse position or
 * IS_FIRST_INVOCATION in compute shaders.
 *
 * NOTE: Floating point representation might not be very precise (see print_value(float)).
 *
 * IMPORTANT: Multipler drawcalls can write to the buffer in sequence (if they are from different
 * shgroups). However, we add barriers to support this case and it might change the application
 * behavior. Uncomment DISABLE_DEBUG_SHADER_PRINT_BARRIER to remove the barriers if that happens.
 * But then you are limited to a single invocation output.
 */

#ifndef DRAW_DEBUG_PRINT_DATA
/* Comment that out if you want to debug a library included in multiply shader.
 * Then use DRAW_DEBUG_PRINT_DATA to check if the print buffer is present. */
// #  error "Missing draw_debug_print additional create info on shader create info"
#else

/* Vertex shader discards the first 3 vertex. */
#  define print_row drw_print_buf[0]
#  define print_col drw_print_buf[1]
#  define print_cursor drw_print_buf[2]

#  define DRW_DEBUG_PRINT_MAX 4096

#  define PRINT_WORD_WRAP_COLUMN 80u

void print_newline()
{
  atomicExchange(print_col, 0u);
  atomicAdd(print_row, 1u);
}

void print_string_start(uint len)
{
  /* Break before word. */
  if (print_col + len > PRINT_WORD_WRAP_COLUMN) {
    print_newline();
  }
}

void print_char4(uint data)
{
  /* Convert into char stream. */
  for (; data != 0u; data >>= 8u) {
    uint char1 = data & 0xFFu;
    /* Check for null terminator. */
    if (char1 == 0x00) {
      break;
    }
    uint cursor = atomicAdd(print_cursor, 1u) + 3;
    if (cursor < DRW_DEBUG_PRINT_MAX) {
      /* For future usage. (i.e: Color) */
      uint flags = 0u;
      uint col = atomicAdd(print_col, 1u);
      uint print_header = (flags << 24u) | (print_row << 16u) | (col << 8u);
      drw_print_buf[cursor] = print_header | char1;
      /* Break word. */
      if (print_col > PRINT_WORD_WRAP_COLUMN) {
        print_newline();
      }
    }
  }
}

/**
 * NOTE(fclem): Strange behavior emerge when trying to increment the digit counter inside the
 * append function. It looks like the compiler does not see it is referenced as an index for char4
 * and thus do not capture the right reference. I do not know if this is undefined behavior.
 * As a matter of precaution, we implement all the append function separately.
 * This behavior was observed on both Mesa & amdgpu-pro.
 */
/* Using ascii char code. Expect char1 to be less or equal to 0xFF. Appends chars to the
 * right. */
void print_append_char(uint char1, inout uint char4)
{
  char4 = (char4 << 8u) | char1;
}

void print_append_digit(uint digit, inout uint char4)
{
  char4 = (char4 << 8u) | ((digit > 9u) ? 0x41u - 10u : 0x30u) + digit;
}

void print_append_space(inout uint char4)
{
  char4 = (char4 << 8u) | 0x20u;
}

void print_value_binary(uint value)
{
  print_no_endl("0b");
  print_string_start(10u * 4u);
  uint digits[10] = array(uint)(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  uint digit = 0u;
  for (uint i = 0u; i < 32u; i++) {
    print_append_digit(((value >> i) & 1u), digits[digit / 4u]);
    digit++;
    if ((i % 4u) == 3u) {
      print_append_space(digits[digit / 4u]);
      digit++;
    }
  }
  /* Numbers are written from right to left. So we need to reverse the order. */
  for (int j = 9; j >= 0; j--) {
    print_char4(digits[j]);
  }
}

void print_value_binary(int value)
{
  print_value_binary(uint(value));
}

void print_value_binary(float value)
{
  print_value_binary(floatBitsToUint(value));
}

void print_value_uint(uint value, const bool hex, bool is_negative, const bool is_unsigned)
{
  print_string_start(3u * 4u);
  const uint blank_value = hex ? 0x30303030u : 0x20202020u;
  const uint prefix = hex ? 0x78302020u : 0x20202020u;
  uint digits[3] = array(uint)(blank_value, blank_value, prefix);
  const uint base = hex ? 16u : 10u;
  uint digit = 0u;
  /* Add `u` suffix. */
  if (is_unsigned) {
    print_append_char('u', digits[digit / 4u]);
    digit++;
  }
  /* Number's digits. */
  for (; value != 0u || digit == uint(is_unsigned); value /= base) {
    print_append_digit(value % base, digits[digit / 4u]);
    digit++;
  }
  /* Add negative sign. */
  if (is_negative) {
    print_append_char('-', digits[digit / 4u]);
    digit++;
  }
  /* Need to pad to uint alignment because we are issuing chars in "reverse". */
  for (uint i = digit % 4u; i < 4u && i > 0u; i++) {
    print_append_space(digits[digit / 4u]);
    digit++;
  }
  /* Numbers are written from right to left. So we need to reverse the order. */
  for (int j = 2; j >= 0; j--) {
    print_char4(digits[j]);
  }
}

void print_value_hex(uint value)
{
  print_value_uint(value, true, false, false);
}

void print_value_hex(int value)
{
  print_value_uint(uint(value), true, false, false);
}

void print_value_hex(float value)
{
  print_value_uint(floatBitsToUint(value), true, false, false);
}

void print_value(uint value)
{
  print_value_uint(value, false, false, true);
}

void print_value(int value)
{
  print_value_uint(uint(abs(value)), false, (value < 0), false);
}

void print_value(bool value)
{
  if (value) {
    print_no_endl("true ");
  }
  else {
    print_no_endl("false");
  }
}

/* NOTE(@fclem): This is homebrew and might not be 100% accurate (accuracy has not been tested and
 * might dependent on compiler implementation). If unsure, use print_value_hex and transcribe the
 * value manually with another tool. */
void print_value(float val)
{
  /* We pad the string to match normal float values length. */
  if (isnan(val)) {
    print_no_endl("         NaN");
    return;
  }
  if (isinf(val)) {
    if (sign(val) < 0.0) {
      print_no_endl("        -Inf");
    }
    else {
      print_no_endl("         Inf");
    }
    return;
  }

  /* Adjusted for significant digits (6) with sign (1), decimal separator (1) and exponent (4). */
  const float significant_digits = 6.0;
  print_string_start(3u * 4u);
  uint digits[3] = array(uint)(0x20202020u, 0x20202020u, 0x20202020u);

  float exponent = floor(log(abs(val)) / log(10.0));
  bool display_exponent = exponent >= (significant_digits) ||
                          exponent <= (-significant_digits + 1.0);

  float int_significant_digits = min(exponent + 1.0, significant_digits);
  float dec_significant_digits = max(0.0, significant_digits - int_significant_digits);
  /* Power to get to the rounding point. */
  float rounding_power = dec_significant_digits;

  if (val == 0.0 || isinf(exponent)) {
    display_exponent = false;
    int_significant_digits = dec_significant_digits = 1.0;
  }
  /* Remap to keep significant numbers count. */
  if (display_exponent) {
    int_significant_digits = 1.0;
    dec_significant_digits = significant_digits - int_significant_digits;
    rounding_power = -exponent + dec_significant_digits;
  }
  /* Round at the last significant digit. */
  val = round(val * pow(10.0, rounding_power));
  /* Get back to final exponent. */
  val *= pow(10.0, -dec_significant_digits);

  float int_part;
  float dec_part = modf(val, int_part);

  dec_part *= pow(10.0, dec_significant_digits);

  const uint base = 10u;
  uint digit = 0u;
  /* Exponent */
  uint value = uint(abs(exponent));
  if (display_exponent) {
    for (int i = 0; value != 0u || i == 0; i++, value /= base) {
      print_append_digit(value % base, digits[digit / 4u]);
      digit++;
    }
    /* Exponent sign. */
    uint sign_char = (exponent < 0.0) ? '-' : '+';
    print_append_char(sign_char, digits[digit / 4u]);
    digit++;
    /* Exponent `e` suffix. */
    print_append_char(0x65u, digits[digit / 4u]);
    digit++;
  }
  /* Decimal part. */
  value = uint(abs(dec_part));
#  if 0 /* We don't do that because it makes unstable values really hard to read. */
  /* Trim trailing zeros. */
  while ((value % base) == 0u) {
    value /= base;
    if (value == 0u) {
      break;
    }
  }
#  endif
  if (value != 0u) {
    for (int i = 0; value != 0u || i == 0; i++, value /= base) {
      print_append_digit(value % base, digits[digit / 4u]);
      digit++;
    }
    /* Point separator. */
    print_append_char('.', digits[digit / 4u]);
    digit++;
  }
  /* Integer part. */
  value = uint(abs(int_part));
  for (int i = 0; value != 0u || i == 0; i++, value /= base) {
    print_append_digit(value % base, digits[digit / 4u]);
    digit++;
  }
  /* Negative sign. */
  if (val < 0.0) {
    print_append_char('-', digits[digit / 4u]);
    digit++;
  }
  /* Need to pad to uint alignment because we are issuing chars in "reverse". */
  for (uint i = digit % 4u; i < 4u && i > 0u; i++) {
    print_append_space(digits[digit / 4u]);
    digit++;
  }
  /* Numbers are written from right to left. So we need to reverse the order. */
  for (int j = 2; j >= 0; j--) {
    print_char4(digits[j]);
  }
}

void print_value(vec2 value)
{
  print_no_endl("vec2(", value[0], ", ", value[1], ")");
}

void print_value(vec3 value)
{
  print_no_endl("vec3(", value[0], ", ", value[1], ", ", value[1], ")");
}

void print_value(vec4 value)
{
  print_no_endl("vec4(", value[0], ", ", value[1], ", ", value[2], ", ", value[3], ")");
}

void print_value(ivec2 value)
{
  print_no_endl("ivec2(", value[0], ", ", value[1], ")");
}

void print_value(ivec3 value)
{
  print_no_endl("ivec3(", value[0], ", ", value[1], ", ", value[1], ")");
}

void print_value(ivec4 value)
{
  print_no_endl("ivec4(", value[0], ", ", value[1], ", ", value[2], ", ", value[3], ")");
}

void print_value(uvec2 value)
{
  print_no_endl("uvec2(", value[0], ", ", value[1], ")");
}

void print_value(uvec3 value)
{
  print_no_endl("uvec3(", value[0], ", ", value[1], ", ", value[1], ")");
}

void print_value(uvec4 value)
{
  print_no_endl("uvec4(", value[0], ", ", value[1], ", ", value[2], ", ", value[3], ")");
}

void print_value(bvec2 value)
{
  print_no_endl("bvec2(", value[0], ", ", value[1], ")");
}

void print_value(bvec3 value)
{
  print_no_endl("bvec3(", value[0], ", ", value[1], ", ", value[1], ")");
}

void print_value(bvec4 value)
{
  print_no_endl("bvec4(", value[0], ", ", value[1], ", ", value[2], ", ", value[3], ")");
}

#endif
