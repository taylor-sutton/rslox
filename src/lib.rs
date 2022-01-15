//! The lib crate for a Lox bytecode compiler and interpreter.
#![warn(missing_debug_implementations, missing_docs, rust_2018_idioms)]

/// vm is the bits about running code.
pub mod vm;

/// scanner scans!
pub mod scanner;

/// Takes tokens from the scanner and emits bytecode
pub mod compiler;

mod heap;

/// End-to-end hook-up of the whole intepreter.
/// State is not preserved between calls (yet).
pub fn interpret(src: &str) -> Result<(), vm::LoxError> {
    let s = scanner::Scanner::new(src);
    let c = compiler::compile(s);
    match c {
        None => Err(vm::LoxError::SyntaxError),
        Some((c, h)) => vm::Vm::new_with_heap(c, h).interpret(),
    }
}
