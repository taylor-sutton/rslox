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
#[derive(Debug)]
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
                    (Object::String(s1), Object::String(s2)) => s1 == s2,
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
                Object::String(s) => write!(f, r#""{}""#, s),
            },
        }
    }
}

impl Value {
    fn to_float(&self) -> Result<f64, LoxError> {
        if let Value::Number(f) = self {
            Ok(*f)
        } else {
            Err(LoxError::RuntimeError)
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

pub use heap::Heap;
use heap::*;

// There are more From impls we could imagine, for example, From<Option<T>> where T: Into<Value>
// but we can hold off on them until they are useful

const STACK_SIZE: usize = 256;

/// A Vm is a stateful executor of chunks.
#[derive(Debug)]
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
    heap: Heap,
}

/// Errors that can be returned by running the intepreter.
#[derive(Debug, Clone, Error)]
pub enum LoxError {
    /// SyntaxError is for errors during scanning/parsing
    #[error("syntax error")]
    SyntaxError,
    /// RuntimeError happens with runtime problems, like mismatched types
    #[error("runtime error")]
    RuntimeError,
    /// We have a hardcoded max stack size
    #[error("stack overflow")]
    StackOverflow,
    /// Internal Errors should not occur for code that compiled successfully, but just in case.
    #[error("lox internal error: {0}")]
    InternalError(#[from] InternalError),
}

#[derive(Debug, Clone, Error)]
/// VM error that should never come up in code that compiled correct
pub enum InternalError {
    /// An internal error due to Chunks having a limited number of slots for constants.
    #[error("tried to store more than the maximum number of constants in a chunk")]
    TooManyConstants,
    /// Tried to get the top value from an empty stack
    #[error("popped from an empty stack")]
    EmptyStack,
}

impl Vm {
    /// The VM must be initialized with some code to run.
    pub fn new(chunk: Chunk) -> Self {
        Vm {
            chunk,
            ip: 0,
            stack: Vec::with_capacity(STACK_SIZE),
            heap: Heap::new(),
        }
    }

    /// New Vm from a pre-existing heap.
    pub fn new_with_heap(chunk: Chunk, heap: Heap) -> Self {
        Vm {
            chunk,
            ip: 0,
            stack: Vec::with_capacity(STACK_SIZE),
            heap,
        }
    }

    /// Run the interpreter until code finishes executing or an error occurs.
    pub fn interpret(&mut self) -> Result<(), LoxError> {
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
                    Err(LoxError::RuntimeError)
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
                        let mut s: String = a
                            .map_as_string(|x| x.clone())
                            .ok_or(LoxError::RuntimeError)?;
                        b.map_as_string(|bs| s.push_str(bs))
                            .ok_or(LoxError::RuntimeError)?;
                        let new_value = Value::Object(self.heap.new_string_with_value(&s));
                        self.stack_push(new_value)
                    }
                    _ => Err(LoxError::RuntimeError),
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
            .ok_or_else(|| InternalError::TooManyConstants.into())
    }
}

/// heap is our internal interface for allocating objects that can be tracked and GCed. Currently it does not GC,
/// merely allocates and tracks.
///
/// The entry point is the `Heap` type and its `new_*` methods. These methods allocate a new Lox object and return a reference
/// which can be used to access the object: a `HeapRef`. In the future, the `HeapRef` will kept in an alive, usable state as long as it is
/// reachable from a GC root; for now it just lives as long as the Heap does. The underlying objects are all owned by the Heap, so they will
/// be invalid if he Heap is dropped. It is not undefined behavior to store and access HeapRef that is no longer alive - but it will panic.
/// Because of the shared ownership model, values on the heap must be accessed through special APIs.
///
/// ```rust
///    let mut heap = Heap::new();
///    let string_ref = heap.new_string_with_value("my string");
///    // the `map_as_*` method can access the inner value in a convenient way.
///    string_ref.map_as_string(|s| assert_eq!(s, "my string"));
///    // Alternately, the HeapRef can be converted the long way:
///    let shared_obj: SharedObject = heap_ref.as_obj();
///    match shared_obj.borrow() {
///        Object::String(s) => assert_eq!(s, "my string"),
///    };
/// ```
///
/// Internally, the Heap keeps an ``Rc<RefCell<_>>`` and hands out `Weak<RefCell<_>>`. That way, the heap ref is only alive as long as the heap
/// keeps it alive. The HeapRef uses RefCell's runtime borrow checking, meaning that it can be a runtime panic to use incorrectly.
mod heap {
    use std::cell::RefCell;
    use std::ops::{Deref, DerefMut};
    use std::rc::{Rc, Weak};

    #[derive(Debug)]
    pub enum Object {
        String(String),
    }

    // Internal representation of a heap object. The objects are arranged in a linked list using the `next` field, and the objects
    // are owned by the nodes (with the head node owned by the Heap itself)
    #[derive(Debug)]
    struct HeapNode {
        next: Option<Box<HeapNode>>,
        object: Rc<RefCell<Object>>,
    }

    // We hide the implementation details of the shared ownership + interior mutability behind this newtype.
    // However, the `borrow` and `borrow_mut` methods mirror (and delegate to) the RefCell methods of the same name.
    #[derive(Debug)]
    pub struct SharedObject(Rc<RefCell<Object>>);

    impl SharedObject {
        pub fn borrow(&self) -> impl Deref<Target = Object> + '_ {
            self.0.borrow()
        }

        pub fn borrow_mut(&self) -> impl DerefMut<Target = Object> + '_ {
            self.0.borrow_mut()
        }
    }

    #[derive(Debug, Clone)]
    pub struct HeapRef {
        value: Weak<RefCell<Object>>,
    }

    #[derive(Debug)]
    /// A type for allocating, tracking, and eventually GCing heap-allocated Lox object
    pub struct Heap {
        head: Option<Box<HeapNode>>,
    }

    impl Heap {
        /// A new, empty heap.
        pub fn new() -> Heap {
            Heap { head: None }
        }

        /// Add an empty string to the heap, and return the node by which it can be accessed.
        /// If you want to give the string a value ,us `new_string_with_value`, but this method
        /// doesn't do the String's heap allocation (as it uses `String::new()`).
        // I tried for a while to return a value that has a lifetime tied to lifetime of the heap,
        // but couldn't make it work :(.
        pub fn new_string(&mut self) -> HeapRef {
            let obj_in_rc = Rc::new(RefCell::new(Object::String(String::new())));
            let obj = HeapNode {
                object: obj_in_rc.clone(),
                next: self.head.take(),
            };
            self.head = Some(Box::new(obj));
            HeapRef {
                value: std::rc::Rc::downgrade(&obj_in_rc),
            }
        }

        /// Add an non-empty string to the heap, copying from value, and return the node by which it can be accessed.
        pub fn new_string_with_value(&mut self, value: &str) -> HeapRef {
            let obj_in_rc = Rc::new(RefCell::new(Object::String(String::from(value))));
            let obj = HeapNode {
                object: obj_in_rc.clone(),
                next: self.head.take(),
            };
            self.head = Some(Box::new(obj));
            HeapRef {
                value: std::rc::Rc::downgrade(&obj_in_rc),
            }
        }

        // Print out all the objects on the heap in linked list order for debugging.
        #[allow(dead_code)]
        fn dump(&self) {
            let mut next = &self.head;
            while let Some(node) = next {
                match &*node.object.borrow() {
                    Object::String(s) => {
                        println!("{}", s);
                    }
                }
                next = &node.next
            }
        }
    }

    impl Default for Heap {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Object {
        pub fn is_string(&self) -> bool {
            true
        }

        pub fn as_string(&self) -> Option<&String> {
            let Object::String(s) = self;
            Some(s)
        }

        pub fn as_string_mut(&mut self) -> Option<&mut String> {
            let Object::String(s) = self;
            Some(s)
        }
    }

    impl HeapRef {
        // Convert a ref to an object, panicking if the ref is no longer alive.
        pub fn as_obj(&self) -> SharedObject {
            SharedObject(self.value.upgrade().unwrap())
        }

        // Apply a function to the inner object if it's a string, returning the result if it's a string
        // And none if it isn't a string
        pub fn map_as_string<F, Ret>(&self, f: F) -> Option<Ret>
        where
            F: FnOnce(&String) -> Ret,
        {
            Some(f(self.as_obj().borrow().as_string()?))
        }

        // Same as map_as_string but with mutability.
        pub fn map_as_string_mut<F, Ret>(&self, f: F) -> Option<Ret>
        where
            F: FnOnce(&mut String) -> Ret,
        {
            Some(f(self.as_obj().borrow_mut().as_string_mut()?))
        }
    }

    #[cfg(test)]
    mod test {
        use super::*;
        #[test]
        fn test_heap() {
            let mut heap = Heap::new();
            let node = heap.new_string();
            node.as_obj()
                .borrow_mut()
                .as_string_mut()
                .unwrap()
                .push_str("Goodbye, world");

            heap.new_string_with_value("Hello, world");

            heap.dump();
        }
    }
}
