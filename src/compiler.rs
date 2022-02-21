// The book's presentation of Pratt parsing confused me, so here are some notes I have.
// In particular, the book's use of implicit state stored in the Parser was confusing for me - I couldn't figure out
// the invariant of which tokens were supposed to be in 'current' and 'previous' at different points during the parse.
// I found https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html
// to be a very helpful guide to writing a Pratt parser ( happens to be in Rust, but may
// be helpful to anyone).
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
    heap::{Function, Heap, InternedString},
    scanner::{Token, TokenType},
    vm::{Chunk, Instruction, Value, JUMP_SENTINEL},
};

// Parser takes a source of tokens, and spits out a chunk.
// It writes to stderr on errors. The public API for Parser is the compile() function.
#[derive(Debug)]
struct Parser<'a, 'heap, T> {
    tokens: T,
    current_token: Token<'a>,
    had_error: bool,
    in_panic_mode: bool,
    heap: &'heap mut Heap,
    // If len == 1, we're at the top level, script
    // if len > 1, we're compiling a function.
    functions: Vec<FunctionCompiler<'a>>,
}

#[derive(Debug)]
struct Local<'token> {
    name: Token<'token>,
    depth: Option<usize>,
}

impl<'token> Local<'token> {
    fn depth_or_panic(&self) -> usize {
        self.depth.expect("depth of an uninitialized local")
    }
}

#[derive(Debug)]
struct FunctionCompiler<'tokens> {
    // Book uses a fixed-sized array as a variable-sized array by tracking the number of elements in use
    // Managing the lifetimes of possibly-not-initialized locals in an array is more trouble then it's worth
    // So let's just use a Vec, and add bounds-checking when adding a local to keep the len within MAX_LOCALS
    locals: Vec<Local<'tokens>>,
    current_depth: usize,
    chunk: Chunk,
    arity: usize,
}

impl<'token> Default for FunctionCompiler<'token> {
    fn default() -> Self {
        let mut ret = Self {
            locals: Vec::with_capacity(Self::MAX_LOCALS),
            current_depth: 0,
            chunk: Chunk::default(),
            arity: 0,
        };
        // the first local is reserved for internal use
        ret.add_unitialized_local(Token {
            typ: TokenType::Identifier,
            raw: "".into(),
            line: 0,
        })
        .unwrap();
        ret.initialize_current_local();
        ret
    }
}

enum GetLocalResult {
    Uninitialized,
    NoSuchLocal,
    FoundLocal(u8),
}

impl<'tokens> FunctionCompiler<'tokens> {
    // Unfortunately, usize from u8 is not const (not in stable) so we can't use u8::MAX.into()
    const MAX_LOCALS: usize = 256;

    fn new() -> FunctionCompiler<'tokens> {
        FunctionCompiler::default()
    }

    fn end(self, name: Option<InternedString>) -> Function {
        Function {
            arity: self.arity,
            code: self.chunk,
            name,
        }
    }

    fn write_instruction(&mut self, instruction: Instruction, line: usize) -> usize {
        self.chunk.write_instruction(instruction, line)
    }

    fn begin_scope(&mut self) {
        self.current_depth += 1;
    }

    fn end_scope(&mut self) {
        let current_len = self.locals.len();
        let new_len = self
            .locals
            .iter()
            .enumerate()
            .rev()
            // Can't end scope in the middle of declaring a local, thus it's okay to
            // call depth_or_panic
            .rfind(|(_, local)| local.depth_or_panic() < self.current_depth)
            .map(|(idx, _)| idx + 1)
            .unwrap_or(0);
        self.current_depth -= 1;
        self.locals.resize_with(new_len, || {
            unreachable!("resizing at end_scope locals shouldn't increase")
        });
        for _ in 0..(current_len - new_len) {
            // TODO Line number of these Pops is lost
            self.write_instruction(Instruction::Pop, 0);
        }
    }

    fn has_locals_capacity(&self) -> bool {
        self.locals.len() < Self::MAX_LOCALS
    }

    // Try to add a local with given name, returning an error message on failure
    fn add_unitialized_local(&mut self, name: Token<'tokens>) -> Result<(), &'static str> {
        if !self.has_locals_capacity() {
            return Err("Too many local variables in function.");
        }

        if self
            .locals
            .iter()
            .rev()
            // Can't create a new local *inside* creating a new local since declaration
            // is a statement and we can't have statements inside expressions.
            // Thus, safe to call depth_or_panic()
            .take_while(|local| local.depth_or_panic() == self.current_depth)
            .any(|local| local.name.raw == name.raw)
        {
            return Err("Already a variable with this name in this scope.");
        }

        self.locals.push(Local { name, depth: None });
        Ok(())
    }

    fn initialize_current_local(&mut self) {
        self.locals
            .last_mut()
            .expect("only call initialize_current_local when there is a local")
            .depth = Some(self.current_depth)
    }

    fn get_local_by_name(&self, name: &str) -> GetLocalResult {
        match self
            .locals
            .iter()
            .enumerate()
            .rfind(|(_, local)| local.name.raw == name)
        {
            None => GetLocalResult::NoSuchLocal,
            Some((_, local)) if local.depth.is_none() => GetLocalResult::Uninitialized,
            Some((idx, _)) => GetLocalResult::FoundLocal(
                idx.try_into()
                    .expect("converting usize index of locals vec into u8"),
            ),
        }
    }
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
            TokenType::And => Some(And),
            TokenType::Or => Some(Or),
            _ => None,
        }
    }
}

use precedence::*;

impl<'a, 'heap, T> Parser<'a, 'heap, T>
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

    fn into_function(mut self) -> Function {
        assert!(self.functions.len() == 1);
        self.functions.pop().unwrap().end(None)
    }

    fn declaration(&mut self) {
        let var_line = self.current_token.line;
        if self.match_token(TokenType::Var) {
            self.var_declaration(var_line);
        } else {
            self.statement();
        }
        if self.in_panic_mode {
            self.synchronize()
        }
    }

    fn synchronize(&mut self) {
        self.in_panic_mode = false;
        // If we see semicolon, go one more then stop
        // If we see the start of a statement, or EOF, stop
        // otherwise, advance.
        while match self.current_token.typ {
            TokenType::Semicolon => {
                self.advance();
                false
            }
            TokenType::Class
            | TokenType::For
            | TokenType::Fun
            | TokenType::If
            | TokenType::Print
            | TokenType::Return
            | TokenType::Var
            | TokenType::While
            | TokenType::Eof => false,
            _ => true,
        } {
            self.advance()
        }
    }

    // Assumes current token is an identifer
    fn identifier_constant_at_point(&mut self) -> u8 {
        let heap_ref = self.heap.new_string(self.current_token.raw.to_string());
        self.advance();
        self.current_function()
            .chunk
            .add_constant(Value::Object(heap_ref))
            .expect("TODO too many constants panic")
    }

    fn var_declaration(&mut self, var_line: usize) {
        if self.current_token.typ != TokenType::Identifier {
            self.error("Expecting identifier as variable name.");
            return;
        }

        // If this is a global, it is accessed by name. So the name needs
        // to be saved as a constant with type string, and we need to emit an instruction
        // to define it.
        // OTOH, if it's a local, we just need the name at compile time in the locals array.
        let idx_if_global = if self.current_function().current_depth == 0 {
            Some(self.identifier_constant_at_point())
        } else {
            let local_token = self.current_token.clone();
            self.advance();
            match self.current_function().add_unitialized_local(local_token) {
                Ok(_) => None,
                Err(s) => {
                    self.error(s);
                    return;
                }
            }
        };

        if self.match_token(TokenType::Equal) {
            self.expression();
        } else {
            self.write_instruction(Instruction::Nil, self.current_token.line);
        }
        self.consume(
            TokenType::Semicolon,
            "Expect ';' after variable declaration.",
        );
        if let Some(idx) = idx_if_global {
            self.write_instruction(Instruction::DefineGlobal(idx), var_line);
        } else {
            self.current_function().initialize_current_local();
        }
    }

    fn statement(&mut self) {
        if self.match_token(TokenType::Print) {
            self.print_statement();
        } else if self.match_token(TokenType::LeftBrace) {
            self.current_function().begin_scope();
            self.block();
            self.current_function().end_scope();
        } else if self.match_token(TokenType::If) {
            self.consume(TokenType::LeftParen, "Expect '( after 'if'.");
            self.expression();
            self.consume(TokenType::RightParen, "Expect ') after condition.");
            let then_idx = self.write_instruction_here(Instruction::JumpIfFalse(JUMP_SENTINEL));
            self.write_instruction_here(Instruction::Pop);
            self.statement();
            let else_idx = self.write_instruction_here(Instruction::Jump(JUMP_SENTINEL));
            self.current_function().chunk.patch_jump(then_idx);
            self.write_instruction_here(Instruction::Pop);
            if self.match_token(TokenType::Else) {
                self.statement();
            }
            self.current_function().chunk.patch_jump(else_idx);
        } else if self.match_token(TokenType::While) {
            self.consume(TokenType::LeftParen, "Expect '( after 'if'.");
            let loop_check = self.current_function().chunk.code_len();
            self.expression();
            self.consume(TokenType::RightParen, "Expect ')' after condition.");
            let jump_to_loop_exit =
                self.write_instruction_here(Instruction::JumpIfFalse(JUMP_SENTINEL));
            self.write_instruction_here(Instruction::Pop);
            self.statement();
            let line = self.current_token.line;
            self.current_function().chunk.add_loop_to(loop_check, line);
            self.current_function().chunk.patch_jump(jump_to_loop_exit);
            self.write_instruction_here(Instruction::Pop);
        } else if self.match_token(TokenType::For) {
            self.current_function().begin_scope();
            self.consume(TokenType::LeftParen, "Expect '( after 'for'.");
            if self.match_token(TokenType::Semicolon) {
                // empty initializer is ok
            } else if self.match_token(TokenType::Var) {
                self.var_declaration(self.current_token.line);
            } else {
                self.expression();
            }

            // If there is no increment, the end-of-body loops to here
            // if there is an increment, the end-of-body loops to the increment, and
            // the increment jumps up to here.
            let pre_condition_idx = self.current_function().chunk.code_len();

            let mut end_of_body_jump = pre_condition_idx;

            // condition is optional - leaving it out is an infinite loop
            let condition_jump = if self.match_token(TokenType::Semicolon) {
                None
            } else {
                self.expression();
                self.consume(TokenType::Semicolon, "Expect ';' after loop condition.");
                let jmp = self.write_instruction_here(Instruction::JumpIfFalse(JUMP_SENTINEL));
                self.write_instruction_here(Instruction::Pop);
                Some(jmp)
            };

            // increment (which is optional)
            if !self.match_token(TokenType::RightParen) {
                // The code for the increment goes immediately after the code for the condition
                // (due to simple single-pass nature of our compiler)
                // But it isn't supposed to run then! It's supposed to run after the body, before re-running
                // the condition
                // so we do some jump dancing.

                let jump_to_body = self.write_instruction_here(Instruction::Jump(JUMP_SENTINEL));
                let pre_increment_idx = self.current_function().chunk.code_len();

                // the actual increment
                self.expression();
                // Increment's value isn't used
                self.write_instruction_here(Instruction::Pop);
                self.consume(TokenType::RightParen, "Expect ')' after for clauses.");

                let line = self.current_token.line;
                self.current_function()
                    .chunk
                    .add_loop_to(pre_condition_idx, line);

                end_of_body_jump = pre_increment_idx;

                self.current_function().chunk.patch_jump(jump_to_body);
            }

            self.statement();
            let line = self.current_token.line;
            self.current_function()
                .chunk
                .add_loop_to(end_of_body_jump, line);

            if let Some(jmp) = condition_jump {
                self.current_function().chunk.patch_jump(jmp);
                // condition jumps to here if false, so pop the falsey value off the stack
                self.write_instruction_here(Instruction::Pop);
            }

            self.current_function().end_scope();
        } else {
            self.expression_statement();
        }
    }

    fn print_statement(&mut self) {
        let line = self.current_token.line;
        self.expression();
        self.consume(TokenType::Semicolon, "Expect ';' after value.");
        self.write_instruction(Instruction::Print, line);
    }

    fn block(&mut self) {
        while self.current_token.typ != TokenType::RightBrace
            && self.current_token.typ != TokenType::Eof
        {
            self.declaration();
        }
        self.consume(TokenType::RightBrace, "Expect '}' after block.")
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
        self.expression_with_min_prec(least_precedence(), true);
    }

    // Same contract as expression() regarding parsing an expression and emitting bytecode, but
    // stop if we encounter an operator with precendence strictly less than min.
    // This is the core Pratt loop.
    fn expression_with_min_prec(&mut self, min_precedence: Precedence, allow_assignment: bool) {
        // We expect the first token to be either a prefix operator, or an atom
        let current_line = self.current_token.line;
        match self.current_token.typ {
            TokenType::Number => {
                let number_value = self.current_token.raw.parse().unwrap();
                let idx = self
                    .current_function()
                    .chunk
                    .add_constant(Value::Number(number_value))
                    .expect("adding constant to chunk");
                self.write_instruction(Instruction::Constant(idx), current_line);
                self.advance();
            }
            TokenType::String => {
                let without_quotes = &self.current_token.raw[1..self.current_token.raw.len() - 1];
                let node = self.heap.new_string(without_quotes.into());
                let idx = self
                    .current_function()
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
                self.expression_with_min_prec(least_precedence(), true); // parens reset the precedence and allow assignment

                // the error will be after the expr, which is not ideal, but it's ok
                self.consume(TokenType::RightParen, "Expecting ')' after expression.")
            }
            TokenType::Minus => {
                self.advance();
                self.expression_with_min_prec(Precedence::Unary, false);
                self.write_instruction(Instruction::Negate, current_line);
            }
            TokenType::Bang => {
                self.advance();
                self.expression_with_min_prec(Precedence::Unary, false);
                self.write_instruction(Instruction::Not, current_line);
            }
            TokenType::Identifier => {
                // compiler can't tell that current_function()'s borrow of self can't overlap with this one
                // so do a hacky clone
                let tok = self.current_token.raw.to_owned();
                // NOTE for future self: In the book's solution, this would call variable() which would call namedVariable()
                let (get_instr, set_instr) = match self.current_function().get_local_by_name(&tok) {
                    GetLocalResult::FoundLocal(local_idx) => {
                        self.advance(); // advance over local name, discarding it
                        (
                            Instruction::GetLocal(local_idx),
                            Instruction::SetLocal(local_idx),
                        )
                    }
                    GetLocalResult::NoSuchLocal => {
                        let idx = self.identifier_constant_at_point();
                        (Instruction::GetGlobal(idx), Instruction::SetGlobal(idx))
                    }
                    GetLocalResult::Uninitialized => {
                        self.error("Can't read local variable in its own initializer.");
                        return;
                    }
                };
                match (allow_assignment, self.match_token(TokenType::Equal)) {
                    (true, true) => {
                        self.expression();
                        self.write_instruction(set_instr, current_line);
                        // I don't think an explicit return here makes any difference.
                    }
                    (false, true) => {
                        self.error("Invalid assignment target.");
                        return;
                    }
                    (_, false) => {
                        self.write_instruction(get_instr, current_line);
                    }
                };
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

            // Due to the short-circuiting nature of 'and' and 'or', we need to insert some instructions
            // before adding instructions for the second operand.
            let jump_to_patch = match next_typ {
                TokenType::And => {
                    let end_jump =
                        self.write_instruction_here(Instruction::JumpIfFalse(JUMP_SENTINEL));
                    self.write_instruction_here(Instruction::Pop);
                    Some(end_jump)
                    // the book is written with parsing via recursion, rather than a loop
                    // but we need to do something after the next iteration of hte loop, which is awk.
                }
                TokenType::Or => {
                    let mini_jump =
                        self.write_instruction_here(Instruction::JumpIfFalse(JUMP_SENTINEL));
                    let end_jump = self.write_instruction_here(Instruction::Jump(JUMP_SENTINEL));
                    self.write_instruction_here(Instruction::Pop);
                    self.current_function().chunk.patch_jump(mini_jump);
                    Some(end_jump)
                }
                _ => None,
            };

            // Once we've got the prefix and infix operator part of the expression, and are parsing the second operand,
            // we no longer allow assignment
            self.expression_with_min_prec(prec.next(), false);

            match next_typ {
                TokenType::Minus => {
                    self.write_instruction(Instruction::Subtract, current_line);
                }
                TokenType::Plus => {
                    self.write_instruction(Instruction::Add, current_line);
                }
                TokenType::Slash => {
                    self.write_instruction(Instruction::Divide, current_line);
                }
                TokenType::Star => {
                    self.write_instruction(Instruction::Multiply, current_line);
                }
                TokenType::EqualEqual => {
                    self.write_instruction(Instruction::Equal, current_line);
                }
                TokenType::BangEqual => {
                    self.write_instruction(Instruction::Equal, current_line);
                    self.write_instruction(Instruction::Not, current_line);
                }
                TokenType::Less => {
                    self.write_instruction(Instruction::Less, current_line);
                }
                TokenType::GreaterEqual => {
                    self.write_instruction(Instruction::Less, current_line);
                    self.write_instruction(Instruction::Not, current_line);
                }
                TokenType::Greater => {
                    self.write_instruction(Instruction::Greater, current_line);
                }
                TokenType::LessEqual => {
                    self.write_instruction(Instruction::Greater, current_line);
                    self.write_instruction(Instruction::Not, current_line);
                }
                TokenType::And | TokenType::Or => {
                    // self.compiler.chunk.patch_jump(jump_to_patch.unwrap())
                    self.current_function()
                        .chunk
                        .patch_jump(jump_to_patch.unwrap())
                }
                _ => {
                    self.error("Unepxected token in infix operator position.");
                    break;
                }
            };
        }
    }

    fn current_function(&mut self) -> &mut FunctionCompiler<'a> {
        self.functions.last_mut().unwrap()
    }

    // Convenience wrapper to write an instruction to the chunk with the current line.
    fn write_instruction_here(&mut self, instruction: Instruction) -> usize {
        self.write_instruction(instruction, self.current_token.line)
    }

    fn write_instruction(&mut self, instruction: Instruction, line: usize) -> usize {
        self.current_function().write_instruction(instruction, line)
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
pub fn compile<'a, T>(mut tokens: T, heap: &mut Heap) -> Option<Function>
where
    T: Iterator<Item = Token<'a>>,
{
    let first_token = tokens.next().unwrap();
    let mut parser = Parser {
        tokens,
        current_token: first_token,
        had_error: false,
        in_panic_mode: false,
        heap,
        functions: vec![FunctionCompiler::new()],
    };
    if !parser.compile() {
        Some(parser.into_function())
    } else {
        None
    }
}

#[cfg(test)]
mod test {
    // TODO
    // use super::*;
    // use crate::scanner::Scanner;
    // #[test]
    // fn test_thing() {
    //     let text = "(1 + 3 ) / -(-1 + -2)";
    //     let scanner = Scanner::new(text);
    //     let mut heap = Heap::new();
    //     let chunk = compile(scanner, &mut heap).expect("compiling succeeds");
    //     let debug_text = chunk.disassemble("Test");
    //     println!("{}", debug_text);
    //     // TODO figure out a good API for testing this
    // }
}
