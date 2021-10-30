//! The lib crate for a Lox bytecode compiler and interpreter.
#![warn(missing_debug_implementations, missing_docs, rust_2018_idioms)]

/// vm is the bits about running code.
pub mod vm;

/// scanner scans!
pub mod scanner;

/// Takes tokens from the scanner and emits bytecode
pub mod compiler;
