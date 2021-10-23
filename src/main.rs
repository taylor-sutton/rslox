use rslox::vm::{self, *};

fn main() {
    let mut chunk = Chunk::new();
    let idx = chunk.add_constant(Value::Number(1.2));
    chunk.write_instruction(Instruction::Constant(idx), 1);

    let idx = chunk.add_constant(Value::Number(3.4));
    chunk.write_instruction(Instruction::Constant(idx), 2);

    chunk.write_instruction(Instruction::Add, 3);

    let idx = chunk.add_constant(Value::Number(5.6));
    chunk.write_instruction(Instruction::Constant(idx), 2);

    chunk.write_instruction(Instruction::Divide, 2);

    chunk.write_instruction(Instruction::Negate, 2);
    chunk.write_instruction(Instruction::Return, 3);
    let mut machine = vm::Vm::new(chunk);
    machine.interpret().unwrap();
}
