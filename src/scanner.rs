use std::borrow::Cow;
use std::iter::FusedIterator;

/// Scanner takes in an input and spits out tokens.
#[derive(Debug)]
pub struct Scanner<'a> {
    input: &'a str,
    // rather than having another str to distinguish 'start' from 'current' like the book,
    // we'll just keep track of the number of bytes into the input that we've used up
    // while scanning the next token.
    // that is:
    // - my scanned_input_len is the book's (current - start)
    // - my self.unscanned_input() is the book's current (both are &str)
    // - my self.reset_scanned_input() is the book's `start = current`
    scanned_input_len: usize,
    current_line: usize,
    ended: bool,
}

#[allow(dead_code, missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    // Single-character tokens.
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,
    // One or two character tokens.
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    // Literals.
    Identifier,
    String,
    Number,
    // Keywords.
    And,
    Class,
    Else,
    False,
    For,
    Fun,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,

    Error,
    Eof,
}

/// Token is a single token, including a ref to the raw characters that constitute it
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token<'a> {
    typ: TokenType,
    // The Cow is to handle Error tokens
    // If we are okay to stick to constant errors, those are actually static, so
    // we wouldn't not need the Cow, since 'static: 'a for all 'a.
    // But if we want to mix raw as non-static String errors plus refs to the input, Cow lets us.
    raw: Cow<'a, str>,
    line: usize,
}

impl<'a> Scanner<'a> {
    /// Returns a fresh Scanner, ready to spit out tokens from the given source
    // N.B. adding 'b here means the scanner's input is allowed to have shorter lifetime than the source
    // I dunno if that's useful/does anything, or if it's redundant with compiler's lifetime inference.
    pub fn new<'b>(source: &'b str) -> Scanner<'b>
    where
        'b: 'a,
    {
        Scanner {
            input: source,
            current_line: 1,
            ended: false,
            scanned_input_len: 0,
        }
    }
}

impl<'a> Scanner<'a> {
    /// Returns the next token from the input, advancing the canner.
    /// Errors are represented in-band as TokenType::Error.
    /// The scanner will return one Eof token, then None afterwards.
    pub fn next_token(&mut self) -> Option<Token<'a>> {
        self.skip_whitespace();
        let next_char = match self.take_next_char() {
            None if self.ended => return None,
            None => {
                self.ended = true;
                return Some(Token {
                    typ: TokenType::Eof,
                    raw: Cow::Borrowed(self.input),
                    line: self.current_line,
                });
            }
            Some(c) => c,
        };
        let token = match next_char {
            '(' => self.make_token(TokenType::LeftParen),
            ')' => self.make_token(TokenType::RightParen),
            '{' => self.make_token(TokenType::LeftBrace),
            '}' => self.make_token(TokenType::RightBrace),
            ';' => self.make_token(TokenType::Semicolon),
            ',' => self.make_token(TokenType::Comma),
            '.' => self.make_token(TokenType::Dot),
            '-' => self.make_token(TokenType::Minus),
            '+' => self.make_token(TokenType::Plus),
            '/' => self.make_token(TokenType::Slash),
            '*' => self.make_token(TokenType::Star),
            '!' => {
                if self.take_next_char_if_matches('=') {
                    self.make_token(TokenType::BangEqual)
                } else {
                    self.make_token(TokenType::Bang)
                }
            }
            '=' => {
                if self.take_next_char_if_matches('=') {
                    self.make_token(TokenType::EqualEqual)
                } else {
                    self.make_token(TokenType::Equal)
                }
            }
            '<' => {
                if self.take_next_char_if_matches('=') {
                    self.make_token(TokenType::LessEqual)
                } else {
                    self.make_token(TokenType::Less)
                }
            }
            '>' => {
                if self.take_next_char_if_matches('=') {
                    self.make_token(TokenType::GreaterEqual)
                } else {
                    self.make_token(TokenType::Greater)
                }
            }
            '"' => self.scan_string_literal(),
            '0'..='9' => self.scan_numeric_literal(),
            c if c.is_alphabetic() => self.scan_identifier_or_keyword(),
            c => self.err_token(format!("Unexpected character '{}'.", c)),
        };
        self.reset_scanned_input();
        Some(token)
    }

    fn unscanned_input(&self) -> &'a str {
        if self.scanned_input_len < self.input.len() {
            &self.input[self.scanned_input_len..]
        } else {
            ""
        }
    }

    // Same as book's `peek()`, except it's safe to call even if input is empty, in which case it returns none.
    fn peek_next_char(&mut self) -> Option<char> {
        self.unscanned_input().chars().next()
    }

    // Same as book's `peekNext()` except it's safe to call even if input is empty, in which case it returns none.
    fn peek_next_next_char(&mut self) -> Option<char> {
        self.unscanned_input().chars().nth(1)
    }

    // Same as book's `advance()` except it's safe to call even if input is empty, in which case it returns none.
    fn take_next_char(&mut self) -> Option<char> {
        let next_char = self.peek_next_char()?;
        self.scanned_input_len += next_char.len_utf8();
        Some(next_char)
    }

    fn take_next_char_if_matches(&mut self, target: char) -> bool {
        match self.peek_next_char() {
            None => false,
            Some(c) if c == target => {
                self.scanned_input_len += c.len_utf8();
                true
            }
            Some(_) => false,
        }
    }

    fn skip_whitespace(&mut self) {
        loop {
            match self.peek_next_char() {
                Some('\n') => {
                    self.current_line += 1;
                    self.take_next_char();
                }
                Some(c) if c.is_whitespace() => {
                    // only \n is recognized as newline, no other chars
                    self.take_next_char();
                }
                Some('/') if self.peek_next_next_char() == Some('/') => {
                    // skip to the end of the line, leaving the \n alone, then continue the outer whitespace skipping
                    while self.peek_next_char() != Some('\n') {
                        self.take_next_char();
                    }
                }
                _ => break,
            }
        }
        self.reset_scanned_input();
    }

    // Makes a token of the given type from the scanned portion of input (as determined by scanned_input_len),
    // and the current line.
    // Does NOT reset scanned input, caller of this probably also wants to call that.
    fn make_token(&mut self, typ: TokenType) -> Token<'a> {
        let token = Token {
            typ,
            line: self.current_line,
            raw: Cow::Borrowed(&self.input[0..self.scanned_input_len]),
        };
        token
    }

    // Makes an error token with the given message on the current line.
    fn err_token(&self, message: String) -> Token<'a> {
        Token {
            typ: TokenType::Error,
            raw: Cow::Owned(message),
            line: self.current_line,
        }
    }

    // Equivalent of the book's `current = start;`, in other words, mark the scanned portion of input as done
    // by removing it from input.
    fn reset_scanned_input(&mut self) {
        self.input = self.unscanned_input();
        self.scanned_input_len = 0;
    }

    // Assumes we have just scanned the initial double-quote
    // N.B. Lox does not have escape sequences, in particular double-quote cannot appear in string literals
    // Assuming the input starts at the initial double quote, the token's raw chars will be the chars of the literal,
    // including both the starting and ending quotes.
    fn scan_string_literal(&mut self) -> Token<'a> {
        loop {
            match self.peek_next_char() {
                Some('"') => {
                    return {
                        self.take_next_char();
                        self.make_token(TokenType::String)
                    }
                }
                Some(c) => {
                    if c == '\n' {
                        self.current_line += 1
                    }
                    self.take_next_char();
                }
                None => {
                    // Ran out of input without finding end-quote
                    return self.err_token("Unterminated string literal.".to_string());
                }
            }
        }
    }

    // According to the book, '1.' is not a valid liter, and '1.a' scans as three tokens, one, dot, 'a'.
    // That is, the dot is only brought into the numeric literal token if it's followed by a digit.
    fn scan_numeric_literal(&mut self) -> Token<'a> {
        while let Some('0'..='9') = self.peek_next_char() {
            self.take_next_char();
        }
        if self.peek_next_char() == Some('.')
            && matches!(self.peek_next_next_char(), Some('0'..='9'))
        {
            self.take_next_char(); // decimal
            while let Some('0'..='9') = self.peek_next_char() {
                self.take_next_char();
            }
        }
        self.make_token(TokenType::Number)
    }

    fn scan_identifier_or_keyword(&mut self) -> Token<'a> {
        while self.peek_next_char().map_or(false, |c| c.is_alphanumeric()) {
            self.take_next_char();
        }
        self.make_token(token_type_from_str(&self.input[0..self.scanned_input_len]))
    }
}

// assumes text is not empty
fn token_type_from_str(token_text: &str) -> TokenType {
    let mut chars = token_text.chars();
    match chars.next().unwrap() {
        'a' => keyword_if_equal(&token_text[1..], "nd", TokenType::And),
        'c' => keyword_if_equal(&token_text[1..], "lass", TokenType::Class),
        'e' => keyword_if_equal(&token_text[1..], "lse", TokenType::Else),
        'i' => keyword_if_equal(&token_text[1..], "f", TokenType::If),
        'n' => keyword_if_equal(&token_text[1..], "il", TokenType::Nil),
        'o' => keyword_if_equal(&token_text[1..], "r", TokenType::Or),
        'p' => keyword_if_equal(&token_text[1..], "rint", TokenType::Print),
        'r' => keyword_if_equal(&token_text[1..], "eturn", TokenType::Return),
        's' => keyword_if_equal(&token_text[1..], "uper", TokenType::Super),
        'v' => keyword_if_equal(&token_text[1..], "ar", TokenType::Var),
        'w' => keyword_if_equal(&token_text[1..], "hile", TokenType::While),
        'f' => match chars.next() {
            Some('a') => keyword_if_equal(&token_text[2..], "lse", TokenType::False),
            Some('o') => keyword_if_equal(&token_text[2..], "r", TokenType::For),
            Some('u') => keyword_if_equal(&token_text[2..], "n", TokenType::Fun),
            _ => TokenType::Identifier,
        },
        't' => match chars.next() {
            Some('h') => keyword_if_equal(&token_text[2..], "is", TokenType::This),
            Some('r') => keyword_if_equal(&token_text[2..], "ue", TokenType::True),
            _ => TokenType::Identifier,
        },
        _ => TokenType::Identifier,
    }
}

fn keyword_if_equal(text: &str, keyword_text: &str, typ: TokenType) -> TokenType {
    if text == keyword_text {
        typ
    } else {
        TokenType::Identifier
    }
}

impl<'a> Iterator for Scanner<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

impl<'a> FusedIterator for Scanner<'a> {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn big_happy_path_test() {
        let input = r#"
( // comment
) ( { != == = = ! = /
123.1= /123
"#;
        let scanner = Scanner::new(input);
        let tokens: Vec<Token<'_>> = scanner.collect();
        let expected_tokens = vec![
            Token {
                typ: TokenType::LeftParen,
                raw: "(".into(),
                line: 2,
            },
            Token {
                typ: TokenType::RightParen,
                raw: ")".into(),
                line: 3,
            },
            Token {
                typ: TokenType::LeftParen,
                raw: "(".into(),
                line: 3,
            },
            Token {
                typ: TokenType::LeftBrace,
                raw: "{".into(),
                line: 3,
            },
            Token {
                typ: TokenType::BangEqual,
                raw: "!=".into(),
                line: 3,
            },
            Token {
                typ: TokenType::EqualEqual,
                raw: "==".into(),
                line: 3,
            },
            Token {
                typ: TokenType::Equal,
                raw: "=".into(),
                line: 3,
            },
            Token {
                typ: TokenType::Equal,
                raw: "=".into(),
                line: 3,
            },
            Token {
                typ: TokenType::Bang,
                raw: "!".into(),
                line: 3,
            },
            Token {
                typ: TokenType::Equal,
                raw: "=".into(),
                line: 3,
            },
            Token {
                typ: TokenType::Slash,
                raw: "/".into(),
                line: 3,
            },
            Token {
                typ: TokenType::Number,
                raw: "123.1".into(),
                line: 4,
            },
            Token {
                typ: TokenType::Equal,
                raw: "=".into(),
                line: 4,
            },
            Token {
                typ: TokenType::Slash,
                raw: "/".into(),
                line: 4,
            },
            Token {
                typ: TokenType::Number,
                raw: "123".into(),
                line: 4,
            },
            Token {
                typ: TokenType::Eof,
                raw: "".into(),
                line: 5,
            },
        ];
        assert_eq!(tokens.len(), expected_tokens.len());
        for (i, (expected, got)) in tokens.into_iter().zip(expected_tokens).enumerate() {
            assert_eq!(expected, got, "on the token number {}", i);
        }
    }

    #[test]
    fn test_keywords_and_identifiers() {
        let text =
            "and class else if nil or print return super var while false for fun true this f t fAlse thIS";
        let expected_tokens: Vec<_> = vec![
            ("and", TokenType::And),
            ("class", TokenType::Class),
            ("else", TokenType::Else),
            ("if", TokenType::If),
            ("nil", TokenType::Nil),
            ("or", TokenType::Or),
            ("print", TokenType::Print),
            ("return", TokenType::Return),
            ("super", TokenType::Super),
            ("var", TokenType::Var),
            ("while", TokenType::While),
            ("false", TokenType::False),
            ("for", TokenType::For),
            ("fun", TokenType::Fun),
            ("true", TokenType::True),
            ("this", TokenType::This),
            ("f", TokenType::Identifier),
            ("t", TokenType::Identifier),
            ("fAlse", TokenType::Identifier),
            ("thIS", TokenType::Identifier),
            ("", TokenType::Eof),
        ]
        .into_iter()
        .map(|(raw, typ)| Token {
            typ,
            raw: raw.into(),
            line: 1,
        })
        .collect();
        let tokens: Vec<_> = Scanner::new(text).collect();
        assert_eq!(tokens.len(), expected_tokens.len());
        for (i, (expected, got)) in expected_tokens.into_iter().zip(tokens).enumerate() {
            assert_eq!(expected, got, "comparing token {}", i);
        }
    }
}
