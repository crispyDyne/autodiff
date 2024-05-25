use autodiff::backwards::RcVar;

fn main() {
    let x = RcVar::new(2.0);
    let y = RcVar::new(3.0);
    let z = &(10.0 * &x) + &y;
    let w = &(10.0 * &x) * &y;

    // Perform the backward pass wrt `z`
    z.backward(); // Trigger the backward pass
    println!("x.grad: {:?}", x.grad.borrow());
    println!("y.grad: {:?}", y.grad.borrow());

    // Reset the gradients
    *x.grad.borrow_mut() = 0.0;
    *y.grad.borrow_mut() = 0.0;

    // Perform the backward pass wrt `w`
    w.backward(); // Trigger the backward pass
    println!("x.grad: {:?}", x.grad.borrow()); // 3.0
    println!("y.grad: {:?}", y.grad.borrow()); // 2.0
}
