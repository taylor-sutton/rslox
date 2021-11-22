use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::process;

use rslox::interpret;

fn main() {
    let args: Vec<_> = env::args().collect();

    if args.len() == 1 {
        repl()
    } else if args.len() == 2 {
        run_file(&args[1])
    } else {
        eprintln!("Usage: rslox [file]");
        process::exit(64);
    }
}

fn repl() {
    let mut buf = String::new();
    loop {
        buf.clear();
        print!("> ");
        io::stdout().flush().unwrap();
        io::stdin().read_line(&mut buf).expect("reading from stdin");
        if buf == "\n" {
            continue;
        } else if buf.is_empty() {
            println!();
            break;
        }
        // TODO Interpret creates a fresh VM on every line, which is not very useful.
        interpret(&buf).unwrap_or_else(|e| println!("Interpret error: {:?}", e));
        io::stdout().flush().unwrap();
    }
}
fn run_file(path: &str) {
    let mut file = File::open(path).expect("opening file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("reading file");
    interpret(&contents).unwrap();
}
