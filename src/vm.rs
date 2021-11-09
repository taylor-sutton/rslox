use std::{
    convert::TryFrom,
    fmt::{Display, Write},
};

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
    Object(HeapNode),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Number(l0), Self::Number(r0)) => l0 == r0,
            (Self::Boolean(l0), Self::Boolean(r0)) => l0 == r0,
            (Self::Nil, Self::Nil) => true,
            (Self::Object(l0), Self::Object(r0)) => match (l0.typ(), r0.typ()) {
                (ObjectType::String, ObjectType::String) => l0.as_string() == r0.as_string(),
            },
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
            Self::Object(o) => match o.typ() {
                ObjectType::String => write!(f, "{}", o.as_string().unwrap()),
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

/// A heap for Lox values.
///
/// All heap-allocated Lox structures are owned by a heap, which has methods for allocating new
/// objects and returns types that provide references to those objects.
/// Following the book, we use a linked list of blobs, where each blob has a pointer to the next object,
/// A type tag, and some data that depends on its Lox type.
/// Internally, Heap is *very much unsafe*. We use std::alloc::alloc to allocate memory for heap objects,
/// Then cast parts of those objects to and from Rust types to use them.
/// I spent some time trying to work with boxes and dynamically sized types instead of raw pointers,
/// but could not make it work. OTOH, std::alloc::Layout lets us write and read data in specific layouts as we want.
///
/// TODO The current API is not great, since the HeapNodes are not tied to the Heap they were allocated on
/// and therefore could be dangling if the heap is dropped.
/// Perhaps returning shared HeapNodes as shared refs with a lifetime tied to the heap and using explicit
/// interior mutability would be better.
/// If I'm going to keep this API, probably allocating a node should be unsafe, but it'd be better if it weren't.
mod heap {

    use std::alloc::Layout;

    /// A type for allocating, tracking, and eventually GCing heap-allocated Lox object
    #[derive(Debug)]
    pub struct Heap {
        head: Option<HeapNode>,
    }

    // HeapNode upholds the following invariant:
    // - Its pointer points at a non-null pointer
    // - The pointer is valid as long as the heap it was allocated with hasn't been dropped.
    // - A valid ObjectHeader can be read from it (i.e. it was allocated with a layout
    //   beginning with the layout for ObjectHeader, and a valid ObjectHeader has been written there
    // - If the header's Type is ObjectType::String, the entire layout is layout_for_string(), and
    //   a valid String can be read from the string offset.
    // Automatically !Sync and !Send, which is what we want: thread-safety is out of scope for me.
    #[derive(Debug)]
    pub struct HeapNode(*mut u8);

    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    pub enum ObjectType {
        String,
    }

    struct ObjectHeader {
        next: Option<HeapNode>,
        typ: ObjectType,
    }

    impl Default for Heap {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Heap {
        /// A new, empty heap.
        pub fn new() -> Heap {
            Heap { head: None }
        }

        /// Add an empty string to the heap, and return the node by which it can be accessed
        pub fn push_string_node(&mut self) -> HeapNode {
            let (layout, string_offset) = layout_for_string();
            // SAFETY
            // alloc is safe as long as layout is not zero-sized, which it is not.
            let ptr = unsafe { std::alloc::alloc(layout) };
            let header_ptr: *mut ObjectHeader = ptr.cast();
            let header = ObjectHeader {
                next: self.head.take(),
                typ: ObjectType::String,
            };
            // SAFETY
            // write is safe as long has the pointer we are writing to is valid and aligned
            // which it is, as it was just constructed above with a layout that begins with
            // the layout for ObjectHeader.
            // And since we are writing a valid ObjectHeader, it's safe to read an ObjectHeader
            // from this ptr in the future.
            unsafe { header_ptr.write(header) };
            self.head = Some(HeapNode(ptr));
            // SAFETY
            // Adding an offset from Layout::extend to a pointer allocated on that layout
            // is safe.
            let body_ptr: *mut String = unsafe { ptr.add(string_offset) }.cast();
            // This string is getting dropped at the end of this func, which is breaking things
            let s = String::new();
            // SAFETY
            // Similar as the above ptr.add and header_ptr.write.
            unsafe { body_ptr.write(s) };
            HeapNode(ptr)
        }

        // Pops and returns head of heap, with Next set to None.
        // Helper function for cleaning up the heap
        // We could drop using a normal iteration over the linked list, but it's awkward
        // because we need hold the ref to the next node, THEN dealloc the current node
        // We can do this more safely.
        fn pop(&mut self) -> Option<HeapNode> {
            match &mut self.head {
                None => None,
                Some(node) => {
                    let second_ptr = node.header_mut().next.as_ref().map(|x| HeapNode(x.0));
                    let mut ret = self.head.take();
                    self.head = second_ptr;
                    if let Some(hdr) = ret.as_mut().map(|x| x.header_mut()) {
                        hdr.next = None;
                    }
                    ret
                }
            }
        }

        #[allow(dead_code)]
        fn dump(&self) {
            let mut next = &self.head;
            while let Some(node) = next {
                let hdr = node.header();
                match hdr.typ {
                    ObjectType::String => {
                        let s = node.as_string().unwrap();
                        println!("{}", s);
                    }
                }
                next = &node.header().next;
            }
        }
    }

    impl Drop for Heap {
        fn drop(&mut self) {
            while let Some(node) = self.pop() {
                let hdr = node.header();
                let layout: Layout;
                match hdr.typ {
                    ObjectType::String => {
                        let (l, offset) = layout_for_string();
                        layout = l;
                        // Make sure the String's destructor runs.
                        // SAFETY
                        // HeapNodes uphold the invariant that if the tag is ObjectType::String,
                        // then the layout is layout_for_string() and a valid string lives
                        // at the string offset.
                        let string_ptr: *mut String = unsafe { node.0.add(offset) }.cast();
                        let _: String = unsafe { string_ptr.read() };
                    }
                }
                unsafe { std::alloc::dealloc(node.0, layout) };
            }
        }
    }

    fn layout_for_string() -> (Layout, usize) {
        let layout = Layout::new::<ObjectHeader>();
        let (layout, string_offset) = layout.extend(Layout::new::<String>()).unwrap();
        let layout = layout.pad_to_align();
        (layout, string_offset)
    }

    impl HeapNode {
        fn header(&self) -> &ObjectHeader {
            let header_ptr: *const ObjectHeader = self.0.cast();
            unsafe { header_ptr.as_ref().unwrap() }
        }

        fn header_mut(&mut self) -> &mut ObjectHeader {
            let header_ptr: *mut ObjectHeader = self.0.cast();
            unsafe { header_ptr.as_mut().unwrap() }
        }

        pub fn typ(&self) -> ObjectType {
            self.header().typ
        }

        pub fn as_string(&self) -> Option<&String> {
            if self.typ() != ObjectType::String {
                return None;
            }
            let (_, offset) = layout_for_string();
            let string_ptr: *mut String = unsafe { self.0.add(offset) }.cast();
            unsafe { string_ptr.as_ref() }
        }

        // This could be a safety issue due to mutable aliasing rules
        // Since HeapNode isn't Copy or Clone, outside of this module should have access to
        // the string refs following normal rules
        // but we have our own refs inside this module, which we need to be careful of when doing GC.
        pub fn as_string_mut(&mut self) -> Option<&mut String> {
            if self.typ() != ObjectType::String {
                return None;
            }
            let (_, offset) = layout_for_string();
            let string_ptr: *mut String = unsafe { self.0.add(offset) }.cast();
            unsafe { string_ptr.as_mut() }
        }

        /// If a HeapNode is cloned, it becomes possible to violate aliasing rules, by turning both the
        /// original and the clone into mutable refs. Thus, clone is unsafe.
        pub unsafe fn clone(&self) -> HeapNode {
            HeapNode(self.0)
        }
    }

    #[cfg(test)]
    mod test {
        use super::*;
        #[test]
        fn test_heap() {
            let mut heap = Heap::new();
            let mut node = heap.push_string_node();
            let value = node.as_string_mut().unwrap();
            value.push_str("Goodbye, world");

            let mut node = heap.push_string_node();
            let value = node.as_string_mut().unwrap();
            value.push_str("Hello, world");

            heap.dump();
        }
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
                    // SAFETY TODO
                    // Right now this us just plain unsound, since if there are multiple
                    // loads of the same object constant, we will violate the safety invariant of
                    // HeapNode::clone
                    Value::Object(o) => Value::Object(unsafe { o.clone() }),
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
