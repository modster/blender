#include "COM_MultiThreadedOperation.h"
#include "COM_ExecutionSystem.h"

namespace blender::compositor {

MultiThreadedOperation::MultiThreadedOperation()
{
  m_num_passes = 1;
  flags.is_fullframe_operation = true;
}

void MultiThreadedOperation::update_memory_buffer(MemoryBuffer *output,
                                                  const rcti &output_rect,
                                                  blender::Span<MemoryBuffer *> inputs,
                                                  ExecutionSystem &exec_system)
{
  for (int current_pass = 0; current_pass < m_num_passes; current_pass++) {
    update_memory_buffer_started(output, output_rect, inputs, exec_system, current_pass);
    exec_system.execute_work(output_rect, [=, &exec_system](const rcti &split_rect) {
      multi_update_memory_buffer(output, split_rect, inputs, exec_system, current_pass);
    });
    update_memory_buffer_finished(output, output_rect, inputs, exec_system, current_pass);
  }
}

}  // namespace blender::compositor
