//! The lib crate for a Lox bytecode compiler and interpreter.
#![warn(missing_debug_implementations, missing_docs, rust_2018_idioms)]

use std::collections::HashMap;

/// vm is the bits about running code.
mod vm;

/// scanner scans!
pub mod scanner;

/// Takes tokens from the scanner and emits bytecode
mod compiler;

mod heap;

/// Convenience function for running code on a fresh interpreter.
pub fn interpret(src: &str) -> Result<(), vm::LoxError> {
    Interpreter::new().interpret(src)
}

/// Interpreter is the main entry point for running Lox code
#[derive(Debug, Default)]
pub struct Interpreter {
    heap: heap::Heap,
    globals: HashMap<String, vm::Value>,
}

impl Interpreter {
    /// Initialize a fresh interpreter, ready to run some code.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run some code in the interpreter, updating global state such as heap/global vars.
    pub fn interpret(&mut self, src: &str) -> Result<(), vm::LoxError> {
        let s = scanner::Scanner::new(src);
        let c = compiler::compile(s, &mut self.heap);
        match c {
            None => Err(vm::LoxError::SyntaxError),
            Some(c) => {
                // c.map_as_function(|f| println!("{}\n", f.chunk.disassemble("DUMP")));
                vm::execute(c, &mut self.heap, &mut self.globals)
            }
        }
    }
}

/*
  Scanner: Takes &str and produces tokens
  Compiler: takes tokens and produces a chunk (or errors); uses a heap
  Vm: executes a chunk against a heap and globals map
  but a VM is for executing chunks, so it requires a chunk; meanwhile a compiler needs a source of tokens
  Maybe we can create an Interpreter to coordinate, and to own the shared state
  it can interpret a source, which scans it into tokens, compiles those tokens into a chunk against the current heap,
    then makes a VM to execute it.

  In this framing, VM is a BytecodeExecutor, really

  the compiler uses the heap to store constants of heap-allocated types, including identifiers which are string constants
  these are represented the same way any other Value is, i.e. there's no StringConstant vs String runtime representation

  In this sense, the compiler and the interpreter are really linked - if a compiler produces a chunk which is interpreted
  against a different heap, it won't work.

  so let's say our whole thing is a VM. You can ingest source code into the VM, which it will create a parser for,
  and the parser will tokenize it. Then, the compiler will compile the tokens into bytecode against the VM's mutable state.
Finally, that bytecode will get executed against the VM's mutable state.
*/
