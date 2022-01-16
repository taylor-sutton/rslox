use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::{Display, Write};

use thiserror::Error;

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
    /// Pop and print the top value on the stack
    Print,
    /// Pop the stack and do nothing with it.
    Pop,
    /// Define a global whose value is popped from the stack, and whose name is the constant identified by the u8.
    DefineGlobal(u8),
    /// Push onto the stack the global whose name is the constant with the given index.
    GetGlobal(u8),
    /// Pop and assign to the global whose name is the constant with the given index.
    SetGlobal(u8),
    /// Push local onto the top of the stack, where the local is on the stack at the index.
    GetLocal(u8),
    /// Set local at given index to the value on top of the stack, keeping the top of the stack as-is.
    SetLocal(u8),
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
    const OP_CODE_PRINT: u8 = 14;
    const OP_CODE_POP: u8 = 15;
    const OP_CODE_DEFINE_GLOBAL: u8 = 16;
    const OP_CODE_GET_GLOBAL: u8 = 17;
    const OP_CODE_SET_GLOBAL: u8 = 18;
    const OP_CODE_GET_LOCAL: u8 = 19;
    const OP_CODE_SET_LOCAL: u8 = 20;

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
            Self::Print => writer.write(&[Instruction::OP_CODE_PRINT]),
            Self::Pop => writer.write(&[Instruction::OP_CODE_POP]),
            Self::DefineGlobal(u) => writer.write(&[Instruction::OP_CODE_DEFINE_GLOBAL, *u]),
            Self::GetGlobal(u) => writer.write(&[Instruction::OP_CODE_GET_GLOBAL, *u]),
            Self::SetGlobal(u) => writer.write(&[Instruction::OP_CODE_SET_GLOBAL, *u]),
            Self::GetLocal(u) => writer.write(&[Instruction::OP_CODE_GET_LOCAL, *u]),
            Self::SetLocal(u) => writer.write(&[Instruction::OP_CODE_SET_LOCAL, *u]),
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
            Instruction::Print => write!(f, "OP_PRINT"),
            Instruction::Pop => write!(f, "OP_POP"),
            Instruction::DefineGlobal(u) => write!(f, "OP_DEFINE_GLOBAL {:4}", u),
            Instruction::GetGlobal(u) => write!(f, "OP_GET_GLOBAL {:4}", u),
            Instruction::SetGlobal(u) => write!(f, "OP_SET_GLOBAL {:4}", u),
            Instruction::GetLocal(u) => write!(f, "OP_GET_LOCAL {:4}", u),
            Instruction::SetLocal(u) => write!(f, "OP_SET_LOCAL {:4}", u),
        }
    }
}

/// A chunk is the unit of execution for the VM.
/// I've translate the book's manually managed (len, capacity, ptr to values) into Vecs, since that's what Vecs are.
#[derive(Debug)]
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
    pub fn add_constant(&mut self, constant: Value) -> Result<u8, LoxError> {
        let idx = u8::try_from(self.constants.len())
            .map_err(|_| LoxError::InternalError(InternalError::TooManyConstants))?;

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
    // used for testing
    #[allow(dead_code)]
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
#[derive(Debug, Clone)]
pub enum Value {
    /// Lox has a single 'number' base type, backed by f64.
    /// Even though pos/neg infinity and NaN are allowed, we make no guarantees about how they work.
    /// In particular, equality and ordering may be broken for those values.
    Number(f64),
    /// Boolean backed by Rust bool,
    Boolean(bool),
    /// Nil is a type and a value in Lox.
    Nil,
    /// Object is a heap-allocated, garbage collected value
    Object(HeapRef),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Number(l0), Self::Number(r0)) => l0 == r0,
            (Self::Boolean(l0), Self::Boolean(r0)) => l0 == r0,
            (Self::Nil, Self::Nil) => true,
            (Self::Object(l0), Self::Object(r0)) => {
                match (&*l0.as_obj().borrow(), &*r0.as_obj().borrow()) {
                    (Object::InternedString(i1), Object::InternedString(i2)) => i1 == i2,
                }
            }
            _ => false,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(val) => write!(f, "{}", val),
            Self::Boolean(b) => write!(f, "{}", b),
            Self::Nil => write!(f, "<nil>"),
            Self::Object(o) => match &*o.as_obj().borrow() {
                Object::InternedString(i) => {
                    write!(f, r#""{}""#, i.as_str())
                }
            },
        }
    }
}

impl Value {
    fn to_float(&self) -> Result<f64, LoxError> {
        if let Value::Number(f) = self {
            Ok(*f)
        } else {
            Err(LoxError::RuntimeError(format!(
                "cannot turn '{:?}' into float",
                self
            )))
        }
    }

    fn is_falsey(&self) -> bool {
        match self {
            Value::Boolean(b) => *b,
            Value::Nil => true,
            _ => false,
        }
    }

    fn as_object(&self) -> Option<SharedObject> {
        if let Value::Object(o) = self {
            Some(o.as_obj())
        } else {
            None
        }
    }

    fn as_heap_ref(&self) -> Option<&HeapRef> {
        if let Value::Object(o) = self {
            Some(o)
        } else {
            None
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

use crate::heap::*;

// There are more From impls we could imagine, for example, From<Option<T>> where T: Into<Value>
// but we can hold off on them until they are useful

const STACK_SIZE: usize = 256;

/// A Vm is a stateful executor of chunks.
#[derive(Debug)]
struct Vm<'data> {
    chunk: Chunk,
    ip: usize,
    // The book uses an array for the stack, which means that the values are stored in-line with the VM,
    // rather than behind a pointer indirection.
    // We have a few options to do that in Rust:
    // - Use an array, but with Option or MaybeUninit to initialize the array, and track the valid
    //   indices on the stack, same as the book.
    // - Use a Vec; now accessing the stack requires goes out-of-line with the VM, but it's simpler
    stack: Vec<Value>,
    globals: &'data mut HashMap<String, Value>,
    heap: &'data mut Heap,
}

/// Errors that can be returned by running the intepreter.
#[derive(Debug, Clone, Error)]
pub enum LoxError {
    /// SyntaxError is for errors during scanning/parsing
    #[error("syntax error")]
    SyntaxError,
    /// RuntimeError happens with runtime problems, like mismatched types
    #[error("runtime error: {0}")]
    RuntimeError(String),
    /// We have a hardcoded max stack size
    #[error("stack overflow")]
    StackOverflow,
    /// Internal Errors should not occur for code that compiled successfully, but just in case.
    #[error("lox internal error: {0}")]
    InternalError(#[from] InternalError),
}

#[derive(Debug, Clone, Error)]
/// VM error that should never come up in code that compiled correctly.
// TODO It's sort of unclear if these should just be panics or not. Maybe they should be.
pub enum InternalError {
    /// An internal error due to Chunks having a limited number of slots for constants.
    #[error("tried to store more than the maximum number of constants in a chunk")]
    TooManyConstants,
    /// Tried to get the top value from an empty stack
    #[error("popped from an empty stack")]
    EmptyStack,
    /// Tried to look up the constant corresponding to an identifier, but it wasn't a string
    #[error("constant for identifier not a string")]
    IdentifierTypeError,
}

impl<'data> Vm<'data> {
    /// The VM must be initialized with some code to run.
    fn new(
        chunk: Chunk,
        heap: &'data mut Heap,
        globals: &'data mut HashMap<String, Value>,
    ) -> Self {
        Vm {
            chunk,
            ip: 0,
            stack: Vec::with_capacity(STACK_SIZE),
            heap,
            globals,
        }
    }

    /// Run the interpreter until code finishes executing or an error occurs.
    fn interpret(&mut self) -> Result<(), LoxError> {
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

    fn execute(&mut self, instruction: &Instruction) -> Result<(), LoxError> {
        match instruction {
            Instruction::Return => {
                let val = self.stack_pop()?;
                println!("{}", val);
                Ok(())
            }
            Instruction::Constant(idx) => {
                let value = match self.chunk.get_constant(*idx) {
                    Value::Object(o) => Value::Object(o.clone()),
                    Value::Number(x) => Value::Number(*x),
                    Value::Boolean(x) => Value::Boolean(*x),
                    Value::Nil => Value::Nil,
                };
                self.stack_push(value)
            }
            Instruction::Nil => self.stack_push(Value::Nil),
            Instruction::False => self.stack_push(Value::Boolean(false)),
            Instruction::True => self.stack_push(Value::Boolean(true)),
            Instruction::Negate => {
                let value = self.stack_pop()?;
                if let Value::Number(number) = value {
                    self.stack_push(Value::Number(-number))
                } else {
                    Err(LoxError::RuntimeError("Negating non-number value".into()))
                }
            }
            Instruction::Not => {
                let value = self.stack_pop()?;
                self.stack_push(Value::Boolean(!value.is_falsey()))
            }
            Instruction::Add => {
                let b = self.stack_pop()?;
                let a = self.stack_pop()?;
                match (a, b) {
                    (Value::Number(a), Value::Number(b)) => self.stack_push(Value::Number(a + b)),
                    (Value::Object(a), Value::Object(b)) => {
                        let mut s: String =
                            a.map_as_string(|x| x.to_string()).ok_or_else(|| {
                                LoxError::RuntimeError("Adding non-string object".into())
                            })?;
                        b.map_as_string(|bs| s.push_str(bs)).ok_or_else(|| {
                            LoxError::RuntimeError("Adding non-string objects".into())
                        })?;
                        let new_value = Value::Object(self.heap.new_string(s));
                        self.stack_push(new_value)
                    }
                    _ => Err(LoxError::RuntimeError(
                        "Adding when not both string or both number.".into(),
                    )),
                }
            }
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
            Instruction::Print => {
                let a = self.stack_pop()?;
                println!("{}", a);
                Ok(())
            }
            Instruction::Pop => self.stack_pop().map(|_| ()),
            Instruction::DefineGlobal(u) => match self.chunk.get_constant(*u) {
                Value::Object(o) => {
                    let value = self.stack_peek()?.clone();
                    if let Some(r) = o.map_as_string(|s| {
                        self.globals.insert(s.to_owned(), value);
                    }) {
                        self.stack_pop().map(|_| ())?;
                        Ok(r)
                    } else {
                        // The constant was an object, but not a string
                        Err(LoxError::RuntimeError("TODO".into()))
                    }
                }
                // The constant wasn't an object
                _ => Err(LoxError::RuntimeError("TODO".into())),
            },
            Instruction::GetGlobal(u) => self
                .chunk
                .get_constant(*u)
                .as_heap_ref()
                .ok_or(LoxError::InternalError(InternalError::IdentifierTypeError))
                .and_then(|r: &HeapRef| -> Result<Value, LoxError> {
                    r.map_as_string(|s| -> Result<Value, LoxError> {
                        self.globals.get(s).cloned().ok_or_else(|| {
                            LoxError::RuntimeError(format!("Undefined global '{}'", s))
                        })
                    }) // Option<Result<Value, Err>>
                    .unwrap_or(Err(LoxError::InternalError(
                        InternalError::IdentifierTypeError,
                    )))
                })
                .and_then(|v: Value| self.stack_push(v)),
            // Below is a more imperative version of the above functional version of GetGlobal
            // Which is better? You decide.
            // If there was a way to combine multiple if let into one (where the inner depends on the outer)
            // I'd like that instead.
            // Instruction::GetGlobal(u) => match self.chunk.get_constant(*u) {
            //     Value::Object(o) => {
            //         if let Some(value) = o.map_as_string(|s| self.globals.get(s).cloned()).flatten()
            //         {
            //             self.stack_push(value)
            //         } else {
            //             // The constant was an object, but not a string.
            //             // Or the global with that name didn't exist.
            //             Err(LoxError::RuntimeError)
            //         }
            //     }
            //     // The constant wasn't an object
            //     _ => Err(LoxError::RuntimeError),
            // },
            // I'm not sure if this is beautiful or horrible.
            // It is, certainly, functional.
            Instruction::SetGlobal(u) => self
                .chunk
                .get_constant(*u)
                .as_object()
                .ok_or(LoxError::InternalError(InternalError::IdentifierTypeError)) // constant wasn't an object
                .and_then(|o| {
                    let name = o
                        .borrow()
                        .as_string()
                        .ok_or(LoxError::InternalError(InternalError::IdentifierTypeError))? // constant wasn't a string
                        .to_string();
                    let value = self.stack_peek()?.clone();
                    self.globals
                        .get_mut(&name)
                        .map(|value_ref: &mut Value| {
                            *value_ref = value;
                        })
                        .ok_or_else(|| {
                            LoxError::RuntimeError(format!(
                                "Assign to undefined variable '{}'",
                                name
                            ))
                        })
                }),
            // TODO these panic on malformed bytecode - is that okay?
            Instruction::GetLocal(u) => self.stack_push(self.stack[usize::from(*u)].clone()),
            Instruction::SetLocal(u) => {
                self.stack[usize::from(*u)] = self.stack_peek()?.clone();
                Ok(())
            }
        }
    }

    fn stack_push(&mut self, value: Value) -> Result<(), LoxError> {
        if self.stack.len() >= STACK_SIZE {
            Err(LoxError::StackOverflow)
        } else {
            self.stack.push(value);
            Ok(())
        }
    }

    fn stack_pop(&mut self) -> Result<Value, LoxError> {
        self.stack
            .pop()
            .ok_or_else(|| InternalError::EmptyStack.into())
    }

    fn stack_peek(&self) -> Result<&Value, LoxError> {
        self.stack
            .last()
            .ok_or_else(|| InternalError::EmptyStack.into())
    }
}

/// Execute the chunk until code finishes executing or an error occurs.
pub fn execute(
    chunk: Chunk,
    heap: &mut Heap,
    globals: &mut HashMap<String, Value>,
) -> Result<(), LoxError> {
    Vm::new(chunk, heap, globals).interpret()
}
