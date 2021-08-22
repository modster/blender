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

#include "FN_multi_function_procedure.hh"

#include "BLI_dot_export.hh"
#include "BLI_stack.hh"

namespace blender::fn {

void MFVariable::set_name(std::string name)
{
  name_ = std::move(name);
}

void MFCallInstruction::set_next(MFInstruction *instruction)
{
  if (next_ != nullptr) {
    next_->prev_.remove_first_occurrence_and_reorder(this);
  }
  if (instruction != nullptr) {
    instruction->prev_.append(this);
  }
  next_ = instruction;
}

void MFCallInstruction::set_param_variable(int param_index, MFVariable *variable)
{
  if (params_[param_index] != nullptr) {
    params_[param_index]->users_.remove_first_occurrence_and_reorder(this);
  }
  if (variable != nullptr) {
    BLI_assert(fn_->param_type(param_index).data_type() == variable->data_type());
    variable->users_.append(this);
  }
  params_[param_index] = variable;
}

void MFCallInstruction::set_params(Span<MFVariable *> variables)
{
  BLI_assert(variables.size() == params_.size());
  for (const int i : variables.index_range()) {
    this->set_param_variable(i, variables[i]);
  }
}

void MFBranchInstruction::set_condition(MFVariable *variable)
{
  if (condition_ != nullptr) {
    condition_->users_.remove_first_occurrence_and_reorder(this);
  }
  if (variable != nullptr) {
    variable->users_.append(this);
  }
  condition_ = variable;
}

void MFBranchInstruction::set_branch_true(MFInstruction *instruction)
{
  if (branch_true_ != nullptr) {
    branch_true_->prev_.remove_first_occurrence_and_reorder(this);
  }
  if (instruction != nullptr) {
    instruction->prev_.append(this);
  }
  branch_true_ = instruction;
}

void MFBranchInstruction::set_branch_false(MFInstruction *instruction)
{
  if (branch_false_ != nullptr) {
    branch_false_->prev_.remove_first_occurrence_and_reorder(this);
  }
  if (instruction != nullptr) {
    instruction->prev_.append(this);
  }
  branch_false_ = instruction;
}

void MFDestructInstruction::set_variable(MFVariable *variable)
{
  if (variable_ != nullptr) {
    variable_->users_.remove_first_occurrence_and_reorder(this);
  }
  if (variable != nullptr) {
    variable->users_.append(this);
  }
  variable_ = variable;
}

void MFDestructInstruction::set_next(MFInstruction *instruction)
{
  if (next_ != nullptr) {
    next_->prev_.remove_first_occurrence_and_reorder(this);
  }
  if (instruction != nullptr) {
    instruction->prev_.append(this);
  }
  next_ = instruction;
}

void MFDummyInstruction::set_next(MFInstruction *instruction)
{
  if (next_ != nullptr) {
    next_->prev_.remove_first_occurrence_and_reorder(this);
  }
  if (instruction != nullptr) {
    instruction->prev_.append(this);
  }
  next_ = instruction;
}

MFVariable &MFProcedure::new_variable(MFDataType data_type, std::string name)
{
  MFVariable &variable = *allocator_.construct<MFVariable>().release();
  variable.name_ = std::move(name);
  variable.data_type_ = data_type;
  variable.id_ = variables_.size();
  variables_.append(&variable);
  return variable;
}

MFCallInstruction &MFProcedure::new_call_instruction(const MultiFunction &fn)
{
  MFCallInstruction &instruction = *allocator_.construct<MFCallInstruction>().release();
  instruction.type_ = MFInstructionType::Call;
  instruction.fn_ = &fn;
  instruction.params_ = allocator_.allocate_array<MFVariable *>(fn.param_amount());
  instruction.params_.fill(nullptr);
  call_instructions_.append(&instruction);
  return instruction;
}

MFBranchInstruction &MFProcedure::new_branch_instruction()
{
  MFBranchInstruction &instruction = *allocator_.construct<MFBranchInstruction>().release();
  instruction.type_ = MFInstructionType::Branch;
  branch_instructions_.append(&instruction);
  return instruction;
}

MFDestructInstruction &MFProcedure::new_destruct_instruction()
{
  MFDestructInstruction &instruction = *allocator_.construct<MFDestructInstruction>().release();
  instruction.type_ = MFInstructionType::Destruct;
  destruct_instructions_.append(&instruction);
  return instruction;
}

MFDummyInstruction &MFProcedure::new_dummy_instruction()
{
  MFDummyInstruction &instruction = *allocator_.construct<MFDummyInstruction>().release();
  instruction.type_ = MFInstructionType::Dummy;
  dummy_instructions_.append(&instruction);
  return instruction;
}

MFReturnInstruction &MFProcedure::new_return_instruction()
{
  MFReturnInstruction &instruction = *allocator_.construct<MFReturnInstruction>().release();
  instruction.type_ = MFInstructionType::Return;
  return_instructions_.append(&instruction);
  return instruction;
}

void MFProcedure::add_parameter(MFParamType::InterfaceType interface_type, MFVariable &variable)
{
  params_.append({interface_type, &variable});
}

void MFProcedure::set_entry(MFInstruction &entry)
{
  entry_ = &entry;
}

void MFProcedure::assert_valid() const
{
  /**
   * - Non parameter variables are destructed.
   * - At every instruction, every variable is either initialized or uninitialized.
   * - Input and mutable parameters of call instructions are initialized.
   * - Condition of branch instruction is initialized.
   * - Output parameters of call instructions are not initialized.
   * - Input parameters are never destructed.
   * - Mutable and output parameteres are initialized on every exit.
   * - No aliasing issues in call instructions (can happen when variable is used more than once).
   */
}

MFProcedure::~MFProcedure()
{
  for (MFCallInstruction *instruction : call_instructions_) {
    instruction->~MFCallInstruction();
  }
  for (MFBranchInstruction *instruction : branch_instructions_) {
    instruction->~MFBranchInstruction();
  }
  for (MFDestructInstruction *instruction : destruct_instructions_) {
    instruction->~MFDestructInstruction();
  }
  for (MFDummyInstruction *instruction : dummy_instructions_) {
    instruction->~MFDummyInstruction();
  }
  for (MFReturnInstruction *instruction : return_instructions_) {
    instruction->~MFReturnInstruction();
  }
  for (MFVariable *variable : variables_) {
    variable->~MFVariable();
  }
}

static std::string optional_variable_to_string(const MFVariable *variable)
{
  if (variable == nullptr) {
    return "<null>";
  }
  std::stringstream ss;
  ss << variable->name() << "$" << variable->id();
  return ss.str();
}

bool MFProcedure::validate() const
{
  if (!this->validate_all_instruction_pointers_set()) {
    return false;
  }
  if (!this->validate_all_params_provided()) {
    return false;
  }
  if (!this->validate_same_variables_in_one_call()) {
    return false;
  }
  if (!this->validate_parameters()) {
    return false;
  }
  if (!this->validate_initialization()) {
    return false;
  }
  return true;
}

bool MFProcedure::validate_all_instruction_pointers_set() const
{
  for (const MFCallInstruction *instruction : call_instructions_) {
    if (instruction->next_ == nullptr) {
      return false;
    }
  }
  for (const MFDestructInstruction *instruction : destruct_instructions_) {
    if (instruction->next_ == nullptr) {
      return false;
    }
  }
  for (const MFBranchInstruction *instruction : branch_instructions_) {
    if (instruction->branch_true_ == nullptr) {
      return false;
    }
    if (instruction->branch_false_ == nullptr) {
      return false;
    }
  }
  for (const MFDummyInstruction *instruction : dummy_instructions_) {
    if (instruction->next_ == nullptr) {
      return false;
    }
  }
  return true;
}

bool MFProcedure::validate_all_params_provided() const
{
  for (const MFCallInstruction *instruction : call_instructions_) {
    for (const MFVariable *variable : instruction->params_) {
      if (variable == nullptr) {
        return false;
      }
    }
  }
  for (const MFBranchInstruction *instruction : branch_instructions_) {
    if (instruction->condition_ == nullptr) {
      return false;
    }
  }
  for (const MFDestructInstruction *instruction : destruct_instructions_) {
    if (instruction->variable_ == nullptr) {
      return false;
    }
  }
  return true;
}

bool MFProcedure::validate_same_variables_in_one_call() const
{
  for (const MFCallInstruction *instruction : call_instructions_) {
    const MultiFunction &fn = *instruction->fn_;
    for (const int param_index : fn.param_indices()) {
      const MFParamType param_type = fn.param_type(param_index);
      const MFVariable *variable = instruction->params_[param_index];
      for (const int other_param_index : fn.param_indices()) {
        if (other_param_index == param_index) {
          continue;
        }
        const MFVariable *other_variable = instruction->params_[other_param_index];
        if (other_variable != variable) {
          continue;
        }
        if (ELEM(param_type.interface_type(), MFParamType::Mutable, MFParamType::Output)) {
          /* When a variable is used as mutable or output parameter, it can only be used once. */
          return false;
        }
        const MFParamType other_param_type = fn.param_type(other_param_index);
        /* A variable is allowed to be used as input more than once. */
        if (other_param_type.interface_type() != MFParamType::Input) {
          return false;
        }
      }
    }
  }
  return true;
}

bool MFProcedure::validate_parameters() const
{
  Set<const MFVariable *> variables;
  for (const MFParameter &param : params_) {
    /* One variable cannot be used as multiple parameters. */
    if (!variables.add(param.variable)) {
      return false;
    }
  }
  return true;
}

bool MFProcedure::validate_initialization() const
{
  /* TODO: Issue warning when it maybe wrongly initialized. */
  for (const MFDestructInstruction *instruction : destruct_instructions_) {
    const MFVariable &variable = *instruction->variable_;
    const InitState state = this->find_initialization_state_before_instruction(*instruction,
                                                                               variable);
    if (!state.can_be_initialized) {
      return false;
    }
  }
  for (const MFBranchInstruction *instruction : branch_instructions_) {
    const MFVariable &variable = *instruction->condition_;
    const InitState state = this->find_initialization_state_before_instruction(*instruction,
                                                                               variable);
    if (!state.can_be_initialized) {
      return false;
    }
  }
  for (const MFCallInstruction *instruction : call_instructions_) {
    const MultiFunction &fn = *instruction->fn_;
    for (const int param_index : fn.param_indices()) {
      const MFParamType param_type = fn.param_type(param_index);
      const MFVariable &variable = *instruction->params_[param_index];
      const InitState state = this->find_initialization_state_before_instruction(*instruction,
                                                                                 variable);
      switch (param_type.interface_type()) {
        case MFParamType::Input:
        case MFParamType::Mutable: {
          if (!state.can_be_initialized) {
            return false;
          }
          break;
        }
        case MFParamType::Output: {
          if (!state.can_be_uninitialized) {
            return false;
          }
          break;
        }
      }
    }
  }
  Set<const MFVariable *> variables_that_should_be_initialized_on_return;
  for (const MFParameter &param : params_) {
    if (ELEM(param.type, MFParamType::Mutable, MFParamType::Output)) {
      variables_that_should_be_initialized_on_return.add_new(param.variable);
    }
  }
  for (const MFReturnInstruction *instruction : return_instructions_) {
    for (const MFVariable *variable : variables_) {
      const InitState init_state = this->find_initialization_state_before_instruction(*instruction,
                                                                                      *variable);
      if (variables_that_should_be_initialized_on_return.contains(variable)) {
        if (!init_state.can_be_initialized) {
          return false;
        }
      }
      else {
        if (!init_state.can_be_uninitialized) {
          return false;
        }
      }
    }
  }
  return true;
}

MFProcedure::InitState MFProcedure::find_initialization_state_before_instruction(
    const MFInstruction &target_instruction, const MFVariable &target_variable) const
{
  InitState state;

  auto check_entry_instruction = [&]() {
    bool caller_initialized_variable = false;
    for (const MFParameter &param : params_) {
      if (param.variable == &target_variable) {
        if (ELEM(param.type, MFParamType::Input, MFParamType::Mutable)) {
          caller_initialized_variable = true;
          break;
        }
      }
    }
    if (caller_initialized_variable) {
      state.can_be_initialized = true;
    }
    else {
      state.can_be_uninitialized = true;
    }
  };

  if (&target_instruction == entry_) {
    check_entry_instruction();
  }

  Set<const MFInstruction *> checked_instructions;
  Stack<const MFInstruction *> instructions_to_check;
  instructions_to_check.push_multiple(target_instruction.prev_);

  while (!instructions_to_check.is_empty()) {
    const MFInstruction &instruction = *instructions_to_check.pop();
    if (!checked_instructions.add(&instruction)) {
      /* Skip if the instruction has been checked already. */
      continue;
    }
    bool state_modified = false;
    switch (instruction.type_) {
      case MFInstructionType::Call: {
        const MFCallInstruction &call_instruction = static_cast<const MFCallInstruction &>(
            instruction);
        const MultiFunction &fn = *call_instruction.fn_;
        for (const int param_index : fn.param_indices()) {
          if (call_instruction.params_[param_index] == &target_variable) {
            const MFParamType param_type = fn.param_type(param_index);
            if (param_type.interface_type() == MFParamType::Output) {
              state.can_be_initialized = true;
              state_modified = true;
              break;
            }
          }
        }
        break;
      }
      case MFInstructionType::Destruct: {
        const MFDestructInstruction &destruct_instruction =
            static_cast<const MFDestructInstruction &>(instruction);
        if (destruct_instruction.variable_ == &target_variable) {
          state.can_be_uninitialized = true;
          state_modified = true;
        }
        break;
      }
      case MFInstructionType::Branch:
      case MFInstructionType::Dummy:
      case MFInstructionType::Return: {
        /* These instruction types don't change the initialization state of variables. */
        break;
      }
    }

    if (!state_modified) {
      if (&instruction == entry_) {
        check_entry_instruction();
      }
      instructions_to_check.push_multiple(instruction.prev_);
    }
  }

  return state;
}

std::string MFProcedure::to_dot() const
{
  dot::DirectedGraph digraph;
  Map<MFInstruction *, dot::Node *> dot_nodes;

  for (MFCallInstruction *instruction : call_instructions_) {
    std::stringstream ss;
    const MultiFunction &fn = instruction->fn();
    ss << fn.name();
    ss << "(";
    for (const int param_index : fn.param_indices()) {
      MFParamType param_type = fn.param_type(param_index);
      switch (param_type.interface_type()) {
        case MFParamType::Input: {
          ss << "in: ";
          break;
        }
        case MFParamType::Output: {
          ss << "out: ";
          break;
        }
        case MFParamType::Mutable: {
          ss << "mut: ";
          break;
        }
      }
      MFVariable *variable = instruction->params()[param_index];
      ss << optional_variable_to_string(variable);
      if (param_index < fn.param_amount() - 1) {
        ss << ", ";
      }
    }
    ss << ")";
    dot::Node &dot_node = digraph.new_node(ss.str());
    dot_node.set_shape(dot::Attr_shape::Rectangle);
    dot_nodes.add_new(instruction, &dot_node);
  }
  for (MFBranchInstruction *instruction : branch_instructions_) {
    MFVariable *variable = instruction->condition();
    std::stringstream ss;
    ss << "Branch: " << optional_variable_to_string(variable);
    dot::Node &dot_node = digraph.new_node(ss.str());
    dot_node.set_shape(dot::Attr_shape::Rectangle);
    dot_nodes.add_new(instruction, &dot_node);
  }
  for (MFDestructInstruction *instruction : destruct_instructions_) {
    MFVariable *variable = instruction->variable();
    std::stringstream ss;
    ss << "Destruct: " << optional_variable_to_string(variable);
    dot::Node &dot_node = digraph.new_node(ss.str());
    dot_node.set_shape(dot::Attr_shape::Rectangle);
    dot_nodes.add_new(instruction, &dot_node);
  }
  for (MFDummyInstruction *instruction : dummy_instructions_) {
    dot::Node &dot_node = digraph.new_node("Dummy");
    dot_node.set_shape(dot::Attr_shape::Rectangle);
    dot_nodes.add_new(instruction, &dot_node);
  }
  for (MFReturnInstruction *instruction : return_instructions_) {
    dot::Node &dot_node = digraph.new_node("");
    dot_node.set_shape(dot::Attr_shape::Circle);
    dot_nodes.add_new(instruction, &dot_node);
  }

  auto create_missing_end_node = [&]() -> dot::Node & {
    dot::Node &node = digraph.new_node("");
    node.set_shape(dot::Attr_shape::Diamond);
    node.set_background_color("red");
    return node;
  };

  auto add_edge_to_instruction_or_end = [&](dot::Node &dot_from, MFInstruction *to) {
    if (to == nullptr) {
      dot::Node &dot_end_node = create_missing_end_node();
      digraph.new_edge(dot_from, dot_end_node);
    }
    else {
      dot::Node &dot_to = *dot_nodes.lookup(to);
      digraph.new_edge(dot_from, dot_to);
    }
  };

  for (MFCallInstruction *instruction : call_instructions_) {
    dot::Node &dot_node = *dot_nodes.lookup(instruction);
    add_edge_to_instruction_or_end(dot_node, instruction->next());
  }
  for (MFBranchInstruction *instruction : branch_instructions_) {
    dot::Node &dot_node = *dot_nodes.lookup(instruction);
    add_edge_to_instruction_or_end(dot_node, instruction->branch_true());
    add_edge_to_instruction_or_end(dot_node, instruction->branch_false());
  }
  for (MFDestructInstruction *instruction : destruct_instructions_) {
    dot::Node &dot_node = *dot_nodes.lookup(instruction);
    add_edge_to_instruction_or_end(dot_node, instruction->next());
  }
  for (MFDummyInstruction *instruction : dummy_instructions_) {
    dot::Node &dot_node = *dot_nodes.lookup(instruction);
    add_edge_to_instruction_or_end(dot_node, instruction->next());
  }

  dot::Node &dot_entry = digraph.new_node("Entry");
  add_edge_to_instruction_or_end(dot_entry, entry_);

  return digraph.to_dot_string();
}

}  // namespace blender::fn
