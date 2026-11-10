use std::io::{self, Write};

fn main() {
    let args =
        minipoa::cli::parse(std::env::args_os().skip(1)).unwrap_or_else(|error| error.exit());
    match minipoa::run(args) {
        Ok(output) => {
            if let Err(error) = io::stdout().lock().write_all(output.as_bytes()) {
                if error.kind() != io::ErrorKind::BrokenPipe {
                    eprintln!("minipoa: cannot write output: {error}");
                    std::process::exit(1);
                }
            }
        }
        Err(error) => {
            eprintln!("minipoa: {error}");
            std::process::exit(1);
        }
    }
}
