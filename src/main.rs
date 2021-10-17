use rslox::vm::{self, *};

fn main() {
    let mut chunk = Chunk::new();
    let idx = chunk.add_constant(Value::Number(1.0));
    chunk.write_instruction(Instruction::Constant(idx), 1);
    chunk.write_instruction(Instruction::Return, 2);
    println!("{}", chunk.disassemble("test"));
    let mut machine = vm::Vm::new(chunk);
    machine.interpret().unwrap();
}
