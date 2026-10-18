//! Conservative demonstration guidance. Commands are TARGET-RELATIVE camera-frame
//! acceleration, never hardware/thruster commands. A vehicle adapter must transform them.
use crate::math::{Pose, V3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Acquire,
    Hold,
    Approach,
    Align,
    Capture,
    Abort,
}
impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Acquire => "acquire",
            Self::Hold => "hold",
            Self::Approach => "approach",
            Self::Align => "align",
            Self::Capture => "capture",
            Self::Abort => "abort",
        }
    }
}
pub struct Input<'a> {
    pub pose: &'a Pose,
    pub velocity: V3,
    pub angular_rate: V3,
    pub valid: bool,
    pub timestamp_s: f64,
    pub sigma_position_m: f64,
    pub in_fov: bool,
}
pub struct Command {
    pub mode: Mode,
    pub relative_acceleration: V3,
    pub reason: &'static str,
}
pub struct Guidance {
    mode: Mode,
    previous: Option<f64>,
}
impl Default for Guidance {
    fn default() -> Self {
        Self {
            mode: Mode::Acquire,
            previous: None,
        }
    }
}
impl Guidance {
    fn command(&mut self, mode: Mode, a: V3, reason: &'static str) -> Command {
        self.mode = mode;
        Command {
            mode,
            relative_acceleration: a,
            reason,
        }
    }
    pub fn update(&mut self, input: Input<'_>) -> Command {
        let p = input.pose.position;
        let finite = p
            .iter()
            .chain(input.velocity.iter())
            .chain(input.angular_rate.iter())
            .chain(input.pose.attitude.coords.iter())
            .all(|x| x.is_finite());
        let clock = input.timestamp_s.is_finite()
            && input.timestamp_s >= 0.0
            && self
                .previous
                .is_none_or(|t| input.timestamp_s > t && input.timestamp_s - t <= 0.1);
        if self.mode == Mode::Abort {
            return self.command(Mode::Abort, V3::new(0.0, 0.0, 0.01), "abort_latched");
        }
        if !finite || !clock {
            return self.command(
                Mode::Abort,
                V3::new(0.0, 0.0, 0.01),
                "invalid_state_or_deadline",
            );
        }
        self.previous = Some(input.timestamp_s);
        if self.mode == Mode::Capture {
            return self.command(Mode::Capture, V3::zeros(), "capture_latched");
        }
        let valid = input.valid
            && input.sigma_position_m.is_finite()
            && (0.0..=0.02).contains(&input.sigma_position_m);
        if !valid {
            if p.z < 0.6 && self.mode != Mode::Acquire {
                return self.command(
                    Mode::Abort,
                    V3::new(0.0, 0.0, 0.01),
                    "navigation_lost_near_target",
                );
            }
            // Brake range closure rather than continue a stale approach.
            let a = V3::new(0.0, 0.0, (-input.velocity.z).clamp(0.0, 0.01));
            return self.command(
                if self.mode == Mode::Acquire {
                    Mode::Acquire
                } else {
                    Mode::Hold
                },
                a,
                "navigation_not_trusted",
            );
        }
        if !input.in_fov || p.z < 0.32 || p.z > 10.0 {
            return self.command(Mode::Abort, V3::new(0.0, 0.0, 0.01), "fov_or_keepout");
        }
        let lateral = V3::new(p.x, p.y, 0.0).norm();
        let aligned = input.pose.attitude.angle() < 0.035 && input.angular_rate.norm() < 0.03;
        if p.z <= 0.36 && lateral < 0.01 && aligned && input.velocity.norm() < 0.01 {
            return self.command(
                Mode::Capture,
                V3::zeros(),
                "virtual_capture_constraints_met",
            );
        }
        let closing_limit = ((p.z - 0.35) * 0.06).clamp(0.005, 0.03);
        if input.velocity.z < -closing_limit - 0.02 {
            return self.command(Mode::Abort, V3::new(0.0, 0.0, 0.01), "excess_closing_speed");
        }
        let corridor = 0.01 + 0.03 * p.z;
        let mode = if !aligned || lateral > corridor {
            Mode::Align
        } else {
            Mode::Approach
        };
        let vz = if mode == Mode::Approach {
            -((p.z - 0.35) * 0.3).clamp(0.0, closing_limit)
        } else {
            0.0
        };
        let mut a = V3::new(
            -0.35 * p.x - 1.2 * input.velocity.x,
            -0.35 * p.y - 1.2 * input.velocity.y,
            1.2 * (vz - input.velocity.z),
        );
        if a.norm() > 0.01 {
            a *= 0.01 / a.norm();
        }
        self.command(
            mode,
            a,
            if aligned {
                "corridor_control"
            } else {
                "wait_for_alignment"
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Q;
    fn input(p: &Pose, t: f64) -> Input<'_> {
        Input {
            pose: p,
            velocity: V3::zeros(),
            angular_rate: V3::zeros(),
            valid: true,
            timestamp_s: t,
            sigma_position_m: 0.005,
            in_fov: true,
        }
    }
    #[test]
    fn loss_brakes_then_aborts_near_target_and_stays_latched() {
        let mut g = Guidance::default();
        let mut p = Pose {
            position: V3::new(0.0, 0.0, 1.0),
            attitude: Q::identity(),
        };
        assert_eq!(g.update(input(&p, 0.0)).mode, Mode::Approach);
        let mut i = input(&p, 0.02);
        i.valid = false;
        i.velocity.z = -0.02;
        let c = g.update(i);
        assert_eq!(c.mode, Mode::Hold);
        assert!(c.relative_acceleration.z > 0.0);
        p.position.z = 0.5;
        let mut i = input(&p, 0.04);
        i.valid = false;
        assert_eq!(g.update(i).mode, Mode::Abort);
        assert_eq!(g.update(input(&p, 0.06)).mode, Mode::Abort);
    }
    #[test]
    fn capture_requires_alignment_and_good_clock() {
        let p = Pose {
            position: V3::new(0.0, 0.0, 0.355),
            attitude: Q::identity(),
        };
        let mut g = Guidance::default();
        assert_eq!(g.update(input(&p, 0.0)).mode, Mode::Capture);
        let mut g = Guidance::default();
        let mut i = input(&p, 0.0);
        i.angular_rate.z = 0.1;
        assert_eq!(g.update(i).mode, Mode::Align);
        assert_eq!(g.update(input(&p, 0.2)).mode, Mode::Abort);
    }
}
