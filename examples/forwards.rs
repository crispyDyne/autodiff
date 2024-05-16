use autodiff::forwards::VarF;

fn main() {
    let x = VarF::new(3.0, 1.0); // x = 3.0
    let y = VarF::new(4.0, 0.0); // y = 4.0

    let z = x * y; // z = x * y = 12.0
    println!("z = {}", z.value); // 12.0
    println!("dz/dx = {}", z.deriv); // 4.0
}
