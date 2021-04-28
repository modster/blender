#include "COM_BufferOperation.h"

namespace blender::compositor {

BufferOperation::BufferOperation(MemoryBuffer *buffer, DataType data_type) : NodeOperation()
{
  m_buffer = buffer;
  unsigned int resolution[2];
  resolution[0] = buffer->getWidth();
  resolution[1] = buffer->getHeight();
  setResolution(resolution);
  addOutputSocket(data_type);
}

void *BufferOperation::initializeTileData(rcti * /*rect*/)
{
  return m_buffer;
}

void BufferOperation::executePixelSampled(float output[4], float x, float y, PixelSampler sampler)
{
  switch (sampler) {
    case PixelSampler::Nearest:
      m_buffer->read(output, x, y);
      break;
    case PixelSampler::Bilinear:
    default:
      m_buffer->readBilinear(output, x, y);
      break;
    case PixelSampler::Bicubic:
      /* No bicubic. Same implementation as ReadBufferOperation. */
      m_buffer->readBilinear(output, x, y);
      break;
  }
}

void BufferOperation::executePixelFiltered(
    float output[4], float x, float y, float dx[2], float dy[2])
{
  const float uv[2] = {x, y};
  const float deriv[2][2] = {{dx[0], dx[1]}, {dy[0], dy[1]}};
  m_buffer->readEWA(output, uv, deriv);
}

}  // namespace blender::compositor
