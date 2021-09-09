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
 */

#include "FN_multi_function_procedure_optimization.hh"

namespace blender::fn::procedure_optimization {

void move_destructs_up(MFProcedure &procedure)
{
  for (MFDestructInstruction *destruct_instr : procedure.destruct_instructions()) {
    MFVariable *variable = destruct_instr->variable();
    if (variable == nullptr) {
      continue;
    }
    if (variable->users().size() != 3) {
      /* Only support the simple case with two uses of the variable for now. */
      continue;
    }
    /* TODO: This is not working yet. */
    MFCallInstruction *last_call_instr = nullptr;
    for (MFInstruction *instr : variable->users()) {
      if (instr->type() == MFInstructionType::Call) {
        MFCallInstruction *call_instr = static_cast<MFCallInstruction *>(instr);
        const int first_param_index = call_instr->params().first_index_try(variable);
        if (call_instr->fn().param_type(first_param_index).interface_type() ==
            MFParamType::Output) {
          last_call_instr = call_instr;
        }
        break;
      }
    }
    if (last_call_instr == nullptr) {
      continue;
    }
    MFInstruction *after_last_call_instr = last_call_instr->next();
    if (after_last_call_instr == destruct_instr) {
      continue;
    }
    if (destruct_instr->prev().size() != 1) {
      continue;
    }
    MFInstruction *before_destruct_instr = destruct_instr->prev()[0];
    MFInstruction *after_destruct_instr = destruct_instr->next();
    if (before_destruct_instr->type() == MFInstructionType::Call) {
      static_cast<MFCallInstruction *>(before_destruct_instr)->set_next(after_destruct_instr);
    }
    else if (before_destruct_instr->type() == MFInstructionType::Destruct) {
      static_cast<MFDestructInstruction *>(before_destruct_instr)->set_next(after_destruct_instr);
    }
    else {
      continue;
    }
    last_call_instr->set_next(destruct_instr);
    destruct_instr->set_next(after_last_call_instr);
  }
}

}  // namespace blender::fn::procedure_optimization
