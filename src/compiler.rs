// The book's presentation of Pratt parsing confused me, so here are some notes I have.
// In particular, the book's use of inplicit state stored in the Parser was confusing for me - I couldn't figure out
// the invariant of which tokens were supposed to be in 'current' and 'previous' at different points during the parse.
// I found https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html
// to be a very helpful guide to writing a Pratt parser ( happens to be in Rust, but may
// be helpuful to anyone).
// In particular, the matklad post illustrates that the parser does not need to be stateful
// (there's no Parser struct in his post, just a parsing function) and that the only API the token source needs
// for the core loop is just tokens.peek(): look at next token, and tokens.next(): move forward one token.
// I've adapted my Parser struct to use this simpler, less stateful interface.
// My parser only stores the first token - the equivalent of tokens.peek(). Storing it in the struct can make
// it a littler easier, so we don't have to pass it as arguments to helper methods all the time.
// And then I store a token source which can give us the next token via next().
// In addition, I've adapted the mapping of tokens to parsing functions and precedences.
// Rather than one monolithic table, we just use Rust's match expression, since what is a table lookup but
// a match expression anyway. It's simpler and more idiomatic Rust to do it with match{}.
// Also, I've used an enum instead of an integer for precedence, because it's more readable and because
// it means all the stuff that would need to change if we introduced more prec levels would be all in one place.
// Using tables is probably a teeny bit faster, but that seems like premature optimization.
// There are crates which provide macros for doing such things with enums, though.

use crate::{
    scanner::{Token, TokenType},
    vm::{Chunk, Heap, Instruction, Value},
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
    heap: Heap,
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
                Top => {
                    // Shrug
                    unreachable!("since no operator has prec of Top, should never call next on Top")
                }
            }
        }
    }

    pub const fn least_precedence() -> Precedence {
        Precedence::Bottom
    }

    use crate::scanner::TokenType;

    pub const fn infix_precedence(typ: &TokenType) -> Option<Precedence> {
        match typ {
            TokenType::Minus => Some(Term),
            TokenType::Plus => Some(Term),
            TokenType::Slash => Some(Factor),
            TokenType::Star => Some(Factor),
            TokenType::EqualEqual => Some(Comparison),
            TokenType::BangEqual => Some(Comparison),
            TokenType::Less => Some(Comparison),
            TokenType::Greater => Some(Comparison),
            TokenType::LessEqual => Some(Comparison),
            TokenType::GreaterEqual => Some(Comparison),
            _ => None,
        }
    }
}

use precedence::*;

impl<'a, T> Parser<'a, T>
where
    T: Iterator<Item = Token<'a>>,
{
    // Our parser is stateful; if you call compile on it, it returns whether the operation succeeded or failed.
    // If it succeeded, the Chunk is available in self.Chunk, otherwise, that chunk may be nonsense bytecode.
    fn compile(&mut self) -> bool {
        while self.current_token.typ != TokenType::Eof {
            self.declaration();
        }

        self.had_error
    }

    fn declaration(&mut self) {
        self.statement()
    }

    fn statement(&mut self) {
        if self.match_token(TokenType::Print) {
            self.print_statement();
        } else {
            self.expression_statement();
        }
    }

    fn print_statement(&mut self) {
        let line = self.current_token.line;
        self.expression();
        self.consume(TokenType::Semicolon, "Expect ';' after value.");
        self.write_instruction(Instruction::Print, line)
    }

    fn expression_statement(&mut self) {
        let line = self.current_token.line;
        self.expression();
        self.consume(TokenType::Semicolon, "Expect ';' after expression.");
        self.write_instruction(Instruction::Pop, line);
    }

    fn match_token(&mut self, typ: TokenType) -> bool {
        if self.current_token.typ == typ {
            self.advance();
            true
        } else {
            false
        }
    }

    // advance is more than just calling next on the token source - it also skips over error tokens.
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

    // Advance if the next token is of the given type, error if the types mismatch
    fn consume(&mut self, expected_type: TokenType, message_if_missing: &str) {
        if self.current_token.typ == expected_type {
            self.advance();
        } else {
            self.error(message_if_missing);
        }
    }

    // The contract of this function is to consume an expression and emit bytecode to the chunk
    // such that the bytecode is a stack-ified verrsion of the expression e.g.
    // if the tokens are 1 + 2, it should emit two constant instructions then an add.
    fn expression(&mut self) {
        self.expression_with_min_prec(least_precedence());
    }

    // Same contract as expression() regarding parsing an expression and emitting bytecode, but
    // stop if we encounter an operator with precendence strictly less than min.
    // This is the core Pratt loop.
    fn expression_with_min_prec(&mut self, min_precedence: Precedence) {
        // We expect the first token to be either a prefix operator, or an atom
        let current_line = self.current_token.line;
        match self.current_token.typ {
            TokenType::Number => {
                let idx = self
                    .chunk
                    .add_constant(Value::Number(self.current_token.raw.parse().unwrap()))
                    .expect("adding constant to chunk");
                self.write_instruction(Instruction::Constant(idx), current_line);
                self.advance();
            }
            TokenType::String => {
                let without_quotes = &self.current_token.raw[1..self.current_token.raw.len() - 1];
                let node = self.heap.new_string(without_quotes.into());
                let idx = self
                    .chunk
                    .add_constant(Value::Object(node))
                    .expect("adding constant to chunk");
                self.write_instruction(Instruction::Constant(idx), current_line);
                self.advance();
            }
            TokenType::Nil => {
                self.write_instruction(Instruction::Nil, current_line);
                self.advance();
            }
            TokenType::True => {
                self.write_instruction(Instruction::True, current_line);
                self.advance();
            }
            TokenType::False => {
                self.write_instruction(Instruction::False, current_line);
                self.advance();
            }
            TokenType::LeftParen => {
                self.advance();
                self.expression_with_min_prec(least_precedence()); // parens reset the precedence

                // the error will be after the expr, which is not ideal, but it's ok
                self.consume(TokenType::RightParen, "Expecting ')' after expression.")
            }
            TokenType::Minus => {
                self.advance();
                self.expression_with_min_prec(Precedence::Unary);
                self.write_instruction(Instruction::Negate, current_line);
            }
            TokenType::Bang => {
                self.advance();
                self.expression_with_min_prec(Precedence::Unary);
                self.write_instruction(Instruction::Not, current_line);
            }

            _ => {
                self.error("Got unexpected token at beginning of expression.");
                return;
            }
        }

        loop {
            let next_typ = self.current_token.typ;
            let current_line = self.current_token.line;

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
                TokenType::Minus => self.write_instruction(Instruction::Subtract, current_line),
                TokenType::Plus => self.write_instruction(Instruction::Add, current_line),
                TokenType::Slash => self.write_instruction(Instruction::Divide, current_line),
                TokenType::Star => self.write_instruction(Instruction::Multiply, current_line),
                TokenType::EqualEqual => self.write_instruction(Instruction::Equal, current_line),
                TokenType::BangEqual => {
                    self.write_instruction(Instruction::Equal, current_line);
                    self.write_instruction(Instruction::Not, current_line);
                }
                TokenType::Less => self.write_instruction(Instruction::Less, current_line),
                TokenType::GreaterEqual => {
                    self.write_instruction(Instruction::Less, current_line);
                    self.write_instruction(Instruction::Not, current_line)
                }
                TokenType::Greater => self.write_instruction(Instruction::Greater, current_line),
                TokenType::LessEqual => {
                    self.write_instruction(Instruction::Greater, current_line);
                    self.write_instruction(Instruction::Not, current_line);
                }
                _ => {
                    self.error("Unepxected token in infix operator position.");
                    break;
                }
            }
        }
    }

    // Convenience wrapper to write an instruction to the chunk with the current line.
    fn write_instruction(&mut self, instruction: Instruction, line: usize) {
        self.chunk.write_instruction(instruction, line)
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
pub fn compile<'a, T>(mut tokens: T) -> Option<(Chunk, Heap)>
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
        heap: Heap::new(),
    };
    if !parser.compile() {
        Some((parser.chunk, parser.heap))
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
        let debug_text = chunk.0.disassemble("Test");
        println!("{}", debug_text);
        // TODO figure out a good API for testing this
    }
}
