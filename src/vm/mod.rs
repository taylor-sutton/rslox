use std::fmt::{Display, Write};

// We have two main options here:
// - Execute based on bytes, skipping the deserialization to a nice struct
// - Deserialize the bytecode into a typesafe struct first
// Not sure which is better - the first is probably faster, but unclear by how much.
// I think this might still be useful for compilation?
#[derive(Debug)]
pub enum Instruction {
    Return,
}

const RETURN_OP_CODE: u8 = 0;

impl Instruction {
    pub fn from_bytes(bytes: &[u8]) -> Option<(Instruction, usize)> {
        if bytes.is_empty() {
            return None;
        }
        match bytes[0] {
            RETURN_OP_CODE => Some((Instruction::Return, 1)),
            _ => None,
        }
    }

    pub fn write_to<W>(&self, writer: &mut W) -> std::io::Result<usize>
    where
        W: std::io::Write,
    {
        match self {
            Self::Return => writer.write(&[RETURN_OP_CODE]),
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Return => write!(f, "OP_RETURN"),
        }
    }
}

// The book uses a size, capacity, and heap-allocated array for chunks, which luckily is what a Vec is
// Similar to our question about Instruction, unclear if this should be Vec<u8> or Vec<Instruction>
#[derive(Debug)]
pub struct Chunk(Vec<u8>);

impl Chunk {
    pub fn new() -> Self {
        Chunk(Vec::new())
    }

    pub fn write_instruction(&mut self, instruction: Instruction) {
        instruction
            .write_to(&mut self.0)
            .expect("writing to vec succeeds");
    }

    pub fn disassemble(&self, title: &str) -> String {
        let mut ret = format!("== {} ==\n", title);
        let mut offset = 0;
        while offset < self.0.len() {
            match Instruction::from_bytes(&self.0[offset..]) {
                Some((i, consumed)) => {
                    write!(&mut ret, "{:04} {}", offset, i)
                        .expect("write to dissamble return string");
                    offset += consumed;
                }
                None => {
                    write!(
                        &mut ret,
                        "skipping unknown instruction at offset {:04}",
                        offset,
                    )
                    .expect("write to dissamble return string");
                    offset += 1;
                }
            }
        }
        ret
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}
