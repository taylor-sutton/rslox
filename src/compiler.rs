// I found https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html
// to e a very helpful guide to writing a Pratt parser in Rust.

use crate::{
    scanner::{Token, TokenType},
    vm::{Chunk, Instruction, Value},
};

// Parser takes a source of tokens, and spits out a chunk.
// It writes to stderr on errors. The public API for Parser is the compile() function.
#[derive(Debug)]
struct Parser<'a, T> {
    tokens: T,
    chunk: Chunk,
    current_token: Token<'a>,
    had_error: bool,
    in_panic_mode: bool,
}

mod precedence {
    // The book uses a C enum. The key, really, is a comparable enum, thus we derive Ord.
    // The book also uses an array indexed by the enums, but that's really just a match statement.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Precedence {
        Bottom,
        Assignment,
        Or,
        And,
        Equality,
        Comparison,
        Term,
        Factor,
        Unary,
        Call,
        Primary,
        Top,
    }
    use Precedence::*;

    impl Precedence {
        pub fn next(&self) -> Precedence {
            match self {
                Bottom => Assignment,
                Assignment => Or,
                Or => And,
                And => Equality,
                Equality => Comparison,
                Comparison => Term,
                Term => Factor,
                Factor => Unary,
                Unary => Call,
                Call => Primary,
                Primary => Top,
                Top => Top,
            }
        }
    }

    pub const fn bottom_precedence() -> Precedence {
        Precedence::Bottom
    }

    use crate::scanner::TokenType;

    pub fn infix_precedence(typ: &TokenType) -> Option<Precedence> {
        match typ {
            TokenType::Minus => Some(Term),
            TokenType::Plus => Some(Term),
            TokenType::Slash => Some(Factor),
            TokenType::Star => Some(Factor),
            _ => None,
        }
    }
}

use precedence::*;

impl<'a, T> Parser<'a, T>
where
    T: Iterator<Item = Token<'a>>,
{
    fn compile(&mut self) -> bool {
        self.expression();
        if self.current_token.typ != TokenType::Eof {
            self.error("Expect EOF at end of expression.");
        }

        self.end_compile();
        self.had_error
    }

    fn advance(&mut self) {
        self.current_token = loop {
            let next = self.tokens.next().expect("TODO handle ran-out-of-tokens");
            if let TokenType::Error = next.typ {
                self.error(&next.raw)
            } else {
                break next;
            }
        };
    }

    fn consume(&mut self, expected_type: TokenType, message_if_missing: &str) {
        if self.current_token.typ == expected_type {
            self.advance();
        } else {
            self.error(message_if_missing);
        }
    }
    fn expression(&mut self) {
        self.expression_with_min_prec(bottom_precedence());
    }

    // The contract of this function is to consume an expression and emit bytecode to the chunk
    // such that the bytecode is a stack-ified verrsion of the expression e.g.
    // if the tokens are 1 + 2, it should emit two constant instructions then an add.
    fn expression_with_min_prec(&mut self, min_precedence: Precedence) {
        // We expect the first token to be either a prefix operator, or an atom
        match self.current_token.typ {
            TokenType::Number => {
                let idx = self
                    .chunk
                    .add_constant(Value::Number(self.current_token.raw.parse().unwrap()))
                    .expect("adding constant to chunk");
                self.write_instruction(Instruction::Constant(idx));
                self.advance();
            }
            TokenType::LeftParen => {
                self.advance();
                self.expression_with_min_prec(bottom_precedence()); // parens reset the precedence
                self.consume(TokenType::RightParen, "Expecting ')' after expression.")
            }
            TokenType::Minus => {
                self.advance();
                self.expression_with_min_prec(Precedence::Unary);
                self.write_instruction(Instruction::Negate);
            }
            _ => {
                self.error("Got unexpected token at beginning of expression.");
                return;
            }
        }

        loop {
            let next_typ = self.current_token.typ;

            let prec = match infix_precedence(&next_typ) {
                Some(prec) => prec,
                None => break,
            };

            if prec < min_precedence {
                break;
            }
            self.advance();
            self.expression_with_min_prec(prec.next());

            match next_typ {
                TokenType::Minus => {
                    self.write_instruction(Instruction::Subtract);
                }
                TokenType::Plus => {
                    self.write_instruction(Instruction::Add);
                }
                TokenType::Slash => {
                    self.write_instruction(Instruction::Divide);
                }
                TokenType::Star => {
                    self.write_instruction(Instruction::Multiply);
                }
                _ => {
                    self.error("Unepxected token in infix operator position.");
                    break;
                }
            }
        }
    }

    fn write_instruction(&mut self, instruction: Instruction) {
        self.chunk
            .write_instruction(instruction, self.current_token.line)
    }

    fn end_compile(&mut self) {
        self.write_instruction(Instruction::Return);
    }

    fn error(&mut self, message: &str) {
        // This clone is a bit ugly, it could be avoided by copying the assignment to had_error into all the
        // erroring fns so that the printing to stderr could be &self instead of &mut self
        let current_token = self.current_token.clone();
        self.error_at(&current_token, message);
    }

    fn error_at(&mut self, token: &Token<'a>, message: &str) {
        if self.in_panic_mode {
            return;
        } else {
            self.in_panic_mode = true;
        }
        eprint!("[line {}] Error", token.line);
        match token.typ {
            TokenType::Eof => eprint!(" at end"),
            TokenType::Error => {}
            _ => eprint!(" at '{}'", token.raw),
        };
        eprintln!(": {}", message);
        self.had_error = true;
    }
}

/// Take a source of tokens, attempt to compile it (writing errors to stderr)
/// and if compilation succeeds, return the chunk.
pub fn compile<'a, T>(mut tokens: T) -> Option<Chunk>
where
    T: Iterator<Item = Token<'a>>,
{
    let first_token = tokens.next().unwrap();
    let mut parser = Parser {
        tokens,
        chunk: Chunk::new(),
        current_token: first_token,
        had_error: false,
        in_panic_mode: false,
    };
    if !parser.compile() {
        Some(parser.chunk)
    } else {
        None
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::scanner::Scanner;

    #[test]
    fn test_thing() {
        let text = "(1 + 3 ) / -(-1 + -2)";
        let scanner = Scanner::new(text);
        let chunk = compile(scanner).expect("compiling succeeds");
        let debug_text = chunk.disassemble("Test");
        println!("{}", debug_text);
        // TODO figure out a good API for testing this
    }
}
