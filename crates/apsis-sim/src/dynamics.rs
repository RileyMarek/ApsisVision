//! Circular-reference-orbit HCW translation and torque-free rigid-body rotation.
//! LVLH axes: X radial outward, Y along-track, Z orbit normal. SI units.
use apsis_math::{Q, V3};
#[derive(Clone)]
pub struct RelativeState {
    pub position: V3,
    pub velocity: V3,
}
impl RelativeState {
    pub fn step(&mut self, dt: f64, n: f64, control: V3) {
        assert!(dt.is_finite() && dt > 0.0 && dt <= 0.1 && n.is_finite() && n >= 0.0);
        fn derivative(p: V3, v: V3, n: f64, a: V3) -> (V3, V3) {
            (
                v,
                V3::new(
                    3.0 * n * n * p.x + 2.0 * n * v.y,
                    -2.0 * n * v.x,
                    -n * n * p.z,
                ) + a,
            )
        }
        let (p, v) = (self.position, self.velocity);
        let k1 = derivative(p, v, n, control);
        let k2 = derivative(p + k1.0 * dt / 2.0, v + k1.1 * dt / 2.0, n, control);
        let k3 = derivative(p + k2.0 * dt / 2.0, v + k2.1 * dt / 2.0, n, control);
        let k4 = derivative(p + k3.0 * dt, v + k3.1 * dt, n, control);
        self.position += (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) * dt / 6.0;
        self.velocity += (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) * dt / 6.0;
    }
}
pub struct RigidBody {
    pub attitude: Q,
    pub rate_body: V3,
    pub inertia: V3,
}
impl RigidBody {
    pub fn step(&mut self, dt: f64) {
        assert!(
            dt.is_finite()
                && dt > 0.0
                && dt <= 0.1
                && self.inertia.iter().all(|x| x.is_finite() && *x > 0.0)
        );
        let f = |w: V3| {
            -w.cross(&self.inertia.component_mul(&w))
                .component_div(&self.inertia)
        };
        let w = self.rate_body;
        let a = f(w);
        let b = f(w + a * dt / 2.0);
        let c = f(w + b * dt / 2.0);
        let d = f(w + c * dt);
        self.rate_body += (a + 2.0 * b + 2.0 * c + d) * dt / 6.0;
        self.attitude *= Q::from_scaled_axis((w + self.rate_body) * dt / 2.0);
    }
}
/// One-dimensional Kelvin-Voigt virtual contact; no latch mechanics or qualification.
pub fn contact_acceleration(range: f64, range_rate: f64) -> f64 {
    if range < 0.35 {
        (200.0 * (0.35 - range) - 20.0 * range_rate).max(0.0) / 12.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hcw_equilibrium_and_ballistic_limit() {
        let mut s = RelativeState {
            position: V3::new(0.0, -2.0, 0.0),
            velocity: V3::zeros(),
        };
        let before = s.position;
        for _ in 0..100 {
            s.step(0.02, 0.0011, V3::zeros());
        }
        assert!((s.position - before).norm() < 1e-12);
        s.velocity.x = 1.0;
        s.step(0.1, 0.0, V3::zeros());
        assert!((s.position.x - 0.1).abs() < 1e-12);
        assert!(contact_acceleration(0.34, -0.01) > 0.0);
        assert_eq!(contact_acceleration(0.36, -0.01), 0.0);
    }
    #[test]
    fn torque_free_energy_and_momentum_are_conserved() {
        let mut b = RigidBody {
            attitude: Q::identity(),
            rate_body: V3::new(0.08, 0.1, 0.03),
            inertia: V3::new(0.002, 0.0004, 0.002),
        };
        let energy = |b: &RigidBody| b.rate_body.dot(&b.inertia.component_mul(&b.rate_body)) / 2.0;
        let e = energy(&b);
        let h = b.attitude * b.inertia.component_mul(&b.rate_body);
        for _ in 0..5000 {
            b.step(0.02);
        }
        assert!((energy(&b) - e).abs() < 1e-10);
        assert!((b.attitude * b.inertia.component_mul(&b.rate_body) - h).norm() < 1e-7);
    }
}
