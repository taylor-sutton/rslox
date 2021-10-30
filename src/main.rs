use std::env;
use std::fs::File;
use std::io::{self, Read};
use std::process;

use rslox::interpret;

fn main() {
    let args: Vec<_> = env::args().collect();
    println!("{:?}", args);

    if args.len() == 1 {
        repl()
    } else if args.len() == 1 {
        run_file(&args[1])
    } else {
        eprintln!("Usage: rslox [file]");
        process::exit(64);
    }
}

fn repl() {
    let mut buf = String::new();
    loop {
        print!("> ");
        io::stdin().read_line(&mut buf).expect("reading from stdin");
        interpret(&buf).unwrap();
        buf.clear();
    }
}
fn run_file(path: &str) {
    let mut file = File::open(path).expect("opening file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("reading file");
    interpret(&contents).unwrap();
}
