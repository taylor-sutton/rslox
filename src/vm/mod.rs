use std::{
    convert::TryFrom,
    fmt::{Display, Write},
};

/// A single instruction, in a parsed/type-safe format.
/// This type is helpful as an intermediate representation while compiling, before serializing into bytes.
/// For execution, we have two options:
/// 1. Parse the bytecode into a Rust type like this, incurring some slowdown while running the VM in exchange
///   for safer and cleaner VM execution loop code.
/// 2. Read the bytecode and execute it directly, for a slight speedup.
/// I'm starting off with choice 1, on the "don't prematurely optimize" guideline, but we'll see
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Instruction {
    /// Return the control flow from the enclosing function.
    Return,
    /// Load a constant by its index into the constant table.
    Constant(u8),
}

const RETURN_OP_CODE: u8 = 0;
const CONSTANT_OP_CODE: u8 = 1;

impl Instruction {
    /// Try to parse an instruction from the beginning of some bytes, returning the number of bytes that the instruction consists of
    /// on success in addition.
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

    /// write_to is a way to get an instruction as bytes in a way that, in some cases, can avoid the extra allocation
    /// that would result from the Into<Vec<u8>> impl
    pub fn write_to<W>(&self, writer: &mut W) -> std::io::Result<usize>
    where
        W: std::io::Write,
    {
        match self {
            Self::Return => writer.write(&[RETURN_OP_CODE]),
            Self::Constant(u) => writer.write(&[CONSTANT_OP_CODE, *u]),
        }
    }

    /// Number of bytes in the byte represention of this instruction
    pub fn num_bytes(&self) -> usize {
        match self {
            Self::Return => 1,
            Self::Constant(_) => 2,
        }
    }
}

impl From<&Instruction> for Vec<u8> {
    fn from(val: &Instruction) -> Self {
        let mut v = vec![];
        val.write_to(&mut v).unwrap();
        v
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

/// A chunk is the unit of execution for the VM.
/// I've translate the book's manually managed (len, capacity, ptr to values) into Vecs, since that's what Vecs are.
#[derive(Debug, Clone)]
pub struct Chunk {
    code: Vec<Instruction>,
    constants: Vec<Value>,
    lines: Vec<usize>,
}

impl Chunk {
    /// A new chunk is empty.
    pub fn new() -> Self {
        Chunk {
            code: Vec::new(),
            constants: Vec::new(),
            lines: Vec::new(),
        }
    }

    /// Add an instruction to the chunk's code.
    pub fn write_instruction(&mut self, instruction: Instruction, line: usize) {
        self.code.push(instruction);
        self.lines.push(line);
    }

    /// Add a constant to the chunk's constants table, returning its index.
    pub fn add_constant(&mut self, constant: Value) -> u8 {
        let idx = match u8::try_from(self.constants.len()) {
            Ok(u) => u,
            Err(_) => panic!("tried to add more than {} constants", u8::MAX), // TODO fix panic
        };

        self.constants.push(constant);
        idx
    }

    /// Return a human-readable string for a chunk.
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

/// VM-internal representation of Lox value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Lox has a single 'number' base type, backed by f64.
    Number(f64),
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(val) => write!(f, "{}", val),
        }
    }
}

/// A Vm is a stateful executor of chunks.
#[derive(Debug, Clone)]
pub struct Vm {
    chunk: Chunk,
    ip: usize,
}

/// Errors that can be returned by running the intepreter.
// TODO impl Error
#[derive(Debug, Clone)]
pub enum InterpretError {
    /// TODO CompileError is returned when...
    CompileError,
    /// TODO RuntimeError is returned when...
    RuntimeError,
}

impl Vm {
    /// The VM must be initialized with some code to run.
    pub fn new(chunk: Chunk) -> Self {
        Vm { chunk, ip: 0 }
    }

    /// Run the interpreter until code finishes executing or an error occurs.
    pub fn interpret(&mut self) -> Result<(), InterpretError> {
        todo!()
    }
}
