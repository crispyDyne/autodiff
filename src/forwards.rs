/// A simple implementation of automatic differentiation using dual numbers.
#[derive(Debug, Clone, Copy)]
pub struct VarF {
    pub value: f64, // The real value
    pub deriv: f64, // The derivative
}

impl VarF {
    /// Constructs a new `AutoDiff`.
    pub fn new(value: f64, deriv: f64) -> Self {
        VarF { value, deriv }
    }
}

// Implementing addition for AutoDiff
use std::ops::Add;

impl Add for VarF {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        VarF {
            value: self.value + other.value,
            deriv: self.deriv + other.deriv,
        }
    }
}

// Implementing multiplication for AutoDiff
use std::ops::Mul;

impl Mul for VarF {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        VarF {
            value: self.value * other.value,
            deriv: self.value * other.deriv + self.deriv * other.value,
        }
    }
}

// Implementing other operations and mathematical functions as needed

use std::ops::{Div, Sub};

impl Sub for VarF {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        VarF {
            value: self.value - other.value,
            deriv: self.deriv - other.deriv,
        }
    }
}

impl Div for VarF {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        VarF {
            value: self.value / other.value,
            deriv: (self.deriv * other.value - self.value * other.deriv)
                / (other.value * other.value),
        }
    }
}
