use rslox::vm::*;

fn main() {
    let mut chunk = Chunk::new();
    chunk.write_instruction(Instruction::Return);
    println!("{:?}", chunk);
    println!("{}", chunk.disassemble("test"))
}
