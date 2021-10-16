use std::{
    convert::TryFrom,
    fmt::{Display, Write},
};

// We have two main options here:
// - Execute based on bytes, skipping the deserialization to a nice struct
// - Deserialize the bytecode into a typesafe struct first
// Not sure which is better - the first is probably faster, but unclear by how much.
// But the latter seems nicer, so I'm going to do that one.
#[derive(Debug)]
pub enum Instruction {
    Return,
    Constant(u8),
}

const RETURN_OP_CODE: u8 = 0;
const CONSTANT_OP_CODE: u8 = 1;

impl Instruction {
    pub fn from_bytes(bytes: &[u8]) -> Option<(Instruction, usize)> {
        if bytes.is_empty() {
            return None;
        }
        match bytes[0] {
            RETURN_OP_CODE => Some((Instruction::Return, 1)),
            CONSTANT_OP_CODE => Some((Instruction::Constant(bytes[1]), 2)), // TODO can panic if OOB
            _ => None,
        }
    }

    pub fn write_to<W>(&self, writer: &mut W) -> std::io::Result<usize>
    where
        W: std::io::Write,
    {
        match self {
            Self::Return => writer.write(&[RETURN_OP_CODE]),
            Self::Constant(u) => writer.write(&[CONSTANT_OP_CODE, *u]),
        }
    }

    pub fn num_bytes(&self) -> usize {
        match self {
            Self::Return => 1,
            Self::Constant(_) => 2,
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Return => write!(f, "OP_RETURN"),
            Instruction::Constant(u) => write!(f, "OP_CONSTANT {:4}", u),
        }
    }
}

// The book uses a size, capacity, and heap-allocated array for chunks, which luckily is what a Vec is
// Similar to our question about Instruction, unclear if this should be Vec<u8> or Vec<Instruction>
#[derive(Debug)]
pub struct Chunk {
    code: Vec<Instruction>,
    constants: Vec<Value>,
    lines: Vec<usize>,
}

impl Chunk {
    pub fn new() -> Self {
        Chunk {
            code: Vec::new(),
            constants: Vec::new(),
            lines: Vec::new(),
        }
    }

    pub fn write_instruction(&mut self, instruction: Instruction, line: usize) {
        self.code.push(instruction);
        self.lines.push(line);
    }

    pub fn add_constant(&mut self, constant: Value) -> u8 {
        let idx = match u8::try_from(self.constants.len()) {
            Ok(u) => u,
            Err(_) => panic!("tried to add more than {} constants", u8::MAX), // TODO fix panic
        };

        self.constants.push(constant);
        idx
    }

    pub fn disassemble(&self, title: &str) -> String {
        let mut ret = format!("== {} ==\n", title);
        let mut offset = 0;
        for (i, instruction) in self.code.iter().enumerate() {
            let line = self.lines[i];
            write!(&mut ret, "{:04} {:04} {}", offset, line, instruction)
                .expect("writing to string");
            #[allow(clippy::single_match)]
            match instruction {
                Instruction::Constant(u) => {
                    write!(
                        &mut ret,
                        " '{}'",
                        self.constants[usize::try_from(*u).unwrap()]
                    )
                    .expect("writing to string");
                }
                _ => {}
            }
            ret.push('\n');
            offset += instruction.num_bytes();
        }
        ret
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum Value {
    Number(f64),
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(val) => write!(f, "{}", val),
        }
    }
}
