use std::{
    convert::TryFrom,
    fmt::{Display, Write},
};

macro_rules! binary_arithmetic {
    ($self:ident, $op:tt) => {
	{
            let b = $self.stack_pop()?.to_float()?;
            let a = $self.stack_pop()?.to_float()?;
	    // Lox is lax about comparing NaNs and stuff
	    #[allow(clippy::float_cmp)]
	    $self.stack_push((a $op b).into())
	}
    };
}

/// A single instruction, in a parsed/type-safe format.
/// This type is helpful as an intermediate representation while compiling, before serializing into bytes.
/// For execution, we have two options:
/// 1. Parse the bytecode into a Rust type like this, incurring some slowdown while running the VM in exchange
///   for safer and cleaner VM execution loop code.
/// 2. Read the bytecode and execute it directly, for a slight speedup.
/// I'm starting off with choice 1, on the "don't prematurely optimize" guideline, but we'll see
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Instruction {
    /// Pop and print the top value
    Return,
    /// Load a constant by its index into the constant table.
    Constant(u8),
    /// Negate the top value on the stack
    Negate,
    /// If stack is TOP: b, a ..., pop two and push (a+b)
    Add,
    /// If stack is TOP: b, a ..., pop two and push (a-b)
    Subtract,
    /// If stack is TOP: b, a ..., pop two and push (a*b)
    Multiply,
    /// If stack is TOP: b, a ..., pop two and push (a/b)
    Divide,
    /// Put Nil on the stack
    Nil,
    /// Put false on the stack
    False,
    /// Put true on the stack
    True,
    /// Logical negation of the top stack item
    Not,
    /// Pop two and push a bool for if they are equal or not
    Equal,
    /// If stack is TOP: b, a, ..., push the bool a>b
    Greater,
    /// If stack is TOP: b, a, ..., push the bool a<b
    Less,
}

impl Instruction {
    const OP_CODE_RETURN: u8 = 0;
    const OP_CODE_CONSTANT: u8 = 1;
    const OP_CODE_NEGATE: u8 = 2;
    const OP_CODE_ADD: u8 = 3;
    const OP_CODE_SUBTRACT: u8 = 4;
    const OP_CODE_MULTIPLY: u8 = 5;
    const OP_CODE_DIVIDE: u8 = 6;
    const OP_CODE_NIL: u8 = 7;
    const OP_CODE_FALSE: u8 = 8;
    const OP_CODE_TRUE: u8 = 9;
    const OP_CODE_NOT: u8 = 10;
    const OP_CODE_EQUAL: u8 = 11;
    const OP_CODE_GREATER: u8 = 12;
    const OP_CODE_LESS: u8 = 13;

    /// Try to parse an instruction from the beginning of some bytes, returning the number of bytes that the instruction consists of
    /// on success in addition.
    pub fn from_bytes(bytes: &[u8]) -> Option<(Instruction, usize)> {
        if bytes.is_empty() {
            return None;
        }
        match bytes[0] {
            Instruction::OP_CODE_RETURN => Some((Instruction::Return, 1)),
            Instruction::OP_CODE_CONSTANT => Some((Instruction::Constant(bytes[1]), 2)), // TODO can panic if OOB
            _ => todo!(),
        }
    }

    /// write_to is a way to get an instruction as bytes in a way that, in some cases, can avoid the extra allocation
    /// that would result from the Into<Vec<u8>> impl
    pub fn write_to<W>(&self, writer: &mut W) -> std::io::Result<usize>
    where
        W: std::io::Write,
    {
        match self {
            Self::Return => writer.write(&[Instruction::OP_CODE_RETURN]),
            Self::Constant(u) => writer.write(&[Instruction::OP_CODE_CONSTANT, *u]),
            Self::Negate => writer.write(&[Instruction::OP_CODE_NEGATE]),
            Self::Not => writer.write(&[Instruction::OP_CODE_NOT]),
            Self::Add => writer.write(&[Instruction::OP_CODE_ADD]),
            Self::Subtract => writer.write(&[Instruction::OP_CODE_SUBTRACT]),
            Self::Multiply => writer.write(&[Instruction::OP_CODE_MULTIPLY]),
            Self::Divide => writer.write(&[Instruction::OP_CODE_DIVIDE]),
            Self::False => writer.write(&[Instruction::OP_CODE_FALSE]),
            Self::True => writer.write(&[Instruction::OP_CODE_TRUE]),
            Self::Nil => writer.write(&[Instruction::OP_CODE_NIL]),
            Self::Equal => writer.write(&[Instruction::OP_CODE_EQUAL]),
            Self::Less => writer.write(&[Instruction::OP_CODE_LESS]),
            Self::Greater => writer.write(&[Instruction::OP_CODE_GREATER]),
        }
    }

    /// Number of bytes in the byte represention of this instruction
    pub fn num_bytes(&self) -> usize {
        match self {
            Instruction::Constant(_) => 2,
            _ => 1,
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
            Instruction::Negate => write!(f, "OP_NEGATE"),
            Instruction::Add => write!(f, "OP_ADD"),
            Instruction::Subtract => write!(f, "OP_SUBTRACT"),
            Instruction::Multiply => write!(f, "OP_MULTIPLY"),
            Instruction::Divide => write!(f, "OP_DIVIDE"),
            Instruction::Nil => write!(f, "OP_NIL"),
            Instruction::False => write!(f, "OP_FALSE"),
            Instruction::True => write!(f, "OP_TRUE"),
            Instruction::Not => write!(f, "OP_NOT"),
            Instruction::Equal => write!(f, "OP_EQUAL"),
            Instruction::Greater => write!(f, "OP_GREATER"),
            Instruction::Less => write!(f, "OP_LESS"),
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
    pub fn add_constant(&mut self, constant: Value) -> Result<u8, InterpretError> {
        let idx =
            u8::try_from(self.constants.len()).map_err(|_| InterpretError::TooManyConstants)?;

        self.constants.push(constant);
        Ok(idx)
    }

    fn get_constant(&self, idx: u8) -> &Value {
        &self.constants[usize::from(idx)]
    }

    fn disassemble_instruction(&self, instruction: &Instruction, ip: usize) -> String {
        let line = self.lines[ip];
        let mut ret = format!("i{:04} {:04} {}", ip, line, instruction);
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
        ret
    }

    /// Return a human-readable string for a chunk.
    pub fn disassemble(&self, title: &str) -> String {
        let mut ret = format!("== {} ==\n", title);
        for (i, instruction) in self.code.iter().enumerate() {
            ret.push_str(&self.disassemble_instruction(instruction, i));
            ret.push('\n');
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
    /// Even though pos/neg infinity and NaN are allowed, we make no guarantees about how they work.
    /// In particular, equality and ordering may be broken for those values.
    Number(f64),
    /// Boolean backed by Rust bool,
    Boolean(bool),
    /// Nil is a type and a value in Lox.
    Nil,
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(val) => write!(f, "{}", val),
            Self::Boolean(b) => write!(f, "{}", b),
            Self::Nil => write!(f, "<nil>"),
        }
    }
}

impl Value {
    fn to_float(&self) -> Result<f64, InterpretError> {
        if let Value::Number(f) = self {
            Ok(*f)
        } else {
            Err(InterpretError::RuntimeError)
        }
    }

    fn to_boolean(&self) -> Result<bool, InterpretError> {
        if let Value::Boolean(b) = self {
            Ok(*b)
        } else {
            Err(InterpretError::RuntimeError)
        }
    }

    fn is_falsey(&self) -> bool {
        match self {
            Value::Boolean(b) => *b,
            Value::Nil => true,
            _ => false,
        }
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Number(f)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Boolean(b)
    }
}

const STACK_SIZE: usize = 256;

/// A Vm is a stateful executor of chunks.
#[derive(Debug, Clone)]
pub struct Vm {
    chunk: Chunk,
    ip: usize,
    // The book uses an array for the stack, which means that the values are stored in-line with the VM,
    // rather than behind a pointer indirection.
    // We have a few options to do that in Rust:
    // - Use an array, but with Option or MaybeUninit to initialize the array, and track the valid
    //   indices on the stack, same as the book.
    // - Use a Vec; now accessing the stack requires goes out-of-line with the VM, but it's simpler
    stack: Vec<Value>,
}

/// Errors that can be returned by running the intepreter.
// TODO impl Error
#[derive(Debug, Clone)]
pub enum InterpretError {
    /// TODO CompileError is returned when...
    CompileError,
    /// TODO RuntimeError is returned when...
    RuntimeError,
    /// An internal error due to Chunks having a limited number of slots for constants.
    TooManyConstants,
}

impl Vm {
    /// The VM must be initialized with some code to run.
    pub fn new(chunk: Chunk) -> Self {
        Vm {
            chunk,
            ip: 0,
            stack: Vec::with_capacity(STACK_SIZE),
        }
    }

    /// Run the interpreter until code finishes executing or an error occurs.
    pub fn interpret(&mut self) -> Result<(), InterpretError> {
        while self.ip < self.chunk.code.len() {
            let instr = self.chunk.code[self.ip];
            #[cfg(feature = "trace")]
            {
                print!("[ ");
                for val in &self.stack {
                    print!("{} ", val)
                }
                println!("]");
                println!("{}", self.chunk.disassemble_instruction(&instr, self.ip));
            }
            self.execute(&instr)?;
            self.ip += 1;
        }
        Ok(())
    }

    fn execute(&mut self, instruction: &Instruction) -> Result<(), InterpretError> {
        match instruction {
            Instruction::Return => {
                let val = self.stack_pop()?;
                println!("{}", val);
                Ok(())
            }
            Instruction::Constant(idx) => self.stack_push(self.chunk.get_constant(*idx).clone()),
            Instruction::Nil => self.stack_push(Value::Nil),
            Instruction::False => self.stack_push(Value::Boolean(false)),
            Instruction::True => self.stack_push(Value::Boolean(true)),
            Instruction::Negate => {
                let value = self.stack_pop()?;
                if let Value::Number(number) = value {
                    self.stack_push(Value::Number(-number))
                } else {
                    Err(InterpretError::RuntimeError)
                }
            }
            Instruction::Not => {
                let value = self.stack_pop()?;
                self.stack_push(Value::Boolean(!value.is_falsey()))
            }
            Instruction::Add => binary_arithmetic!(self, +),
            Instruction::Subtract => binary_arithmetic!(self, -),
            Instruction::Multiply => binary_arithmetic!(self, *),
            Instruction::Divide => binary_arithmetic!(self, /),
            Instruction::Equal => {
                let b = self.stack_pop()?;
                let a = self.stack_pop()?;
                self.stack_push(Value::Boolean(a == b))
            }
            Instruction::Greater => binary_arithmetic!(self, >),
            Instruction::Less => binary_arithmetic!(self, <),
        }
    }

    fn stack_push(&mut self, value: Value) -> Result<(), InterpretError> {
        if self.stack.len() >= STACK_SIZE {
            Err(InterpretError::RuntimeError)
        } else {
            self.stack.push(value);
            Ok(())
        }
    }

    fn stack_pop(&mut self) -> Result<Value, InterpretError> {
        self.stack.pop().ok_or(InterpretError::RuntimeError)
    }
}

// TODO error messages for runtime errors
