use autodiff::backwards::Rev64;
use num_traits::float::Float;

fn main() {
    let x = Rev64::new(2.0); // id = 0
    let y = Rev64::new(3.0); // id = 1
    let a = 10.0; // id = 2
    let b = 10.0; // id = 3
    let mut z = (a * x) + y; // (a * x) id = 4
    let mut w = (b * x) * y; // (b * x) id = 5
    let sin_in = Rev64::new(0.0); // id = 6
    let mut sin = sin_in.sin(); // id = 6

    // Perform the backward pass wrt `z` and print the gradients
    z.backward();
    println!("Gradient of z w.r.t x: {}", x.get_grad());
    println!("Gradient of z w.r.t y: {}", y.get_grad());

    // Perform the backward pass wrt `w` and print the gradients
    w.backward();
    println!("Gradient of w w.r.t x: {}", x.get_grad());
    println!("Gradient of w w.r.t y: {}", y.get_grad());

    // Perform the backward pass wrt `sin_0` and print the gradients
    sin.backward();
    println!("Gradient of sin(0): {}", sin_in.get_grad());
}
