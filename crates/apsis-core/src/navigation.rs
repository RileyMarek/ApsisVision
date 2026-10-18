//! Timestamp and validity contract for a consuming GNC system.
use crate::{math::Pose, Filter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Acquiring,
    Tracking,
    Coasting,
    Lost,
}
impl State {
    pub fn label(self) -> &'static str {
        match self {
            Self::Acquiring => "acquiring",
            Self::Tracking => "tracking",
            Self::Coasting => "coasting",
            Self::Lost => "lost",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    None,
    NoImage,
    Features,
    Association,
    PoseFit,
    Boundary,
    Innovation,
    Stale,
    OutOfOrder,
    InvalidTime,
    Numerical,
    Covariance,
    Reprojection,
    Ambiguous,
}
impl Rejection {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::NoImage => "camera_unavailable",
            Self::Features => "insufficient_features",
            Self::Association => "insufficient_matches",
            Self::PoseFit => "pose_fit_failed",
            Self::Boundary => "unsupported_boundary",
            Self::Innovation => "innovation_gate",
            Self::Stale => "stale_measurement",
            Self::OutOfOrder => "out_of_order",
            Self::InvalidTime => "invalid_timestamp",
            Self::Numerical => "numerical_fault",
            Self::Covariance => "invalid_measurement_covariance",
            Self::Reprojection => "reprojection_gate",
            Self::Ambiguous => "ambiguous_geometry",
        }
    }
}
impl From<crate::vision::integrity::Failure> for Rejection {
    fn from(failure: crate::vision::integrity::Failure) -> Self {
        use crate::vision::integrity::Failure;
        match failure {
            Failure::Fit => Self::PoseFit,
            Failure::Reprojection => Self::Reprojection,
            Failure::Covariance => Self::Covariance,
            Failure::Boundary => Self::Boundary,
            Failure::Ambiguous => Self::Ambiguous,
        }
    }
}
#[derive(Clone, Copy)]
pub struct Policy {
    pub tracking_age_s: f64,
    pub valid_coast_s: f64,
    pub lost_after_s: f64,
    pub max_measurement_age_s: f64,
    pub max_position_sigma_m: f64,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            tracking_age_s: 0.15,
            valid_coast_s: 0.5,
            lost_after_s: 3.5,
            max_measurement_age_s: 0.025,
            max_position_sigma_m: 0.5,
        }
    }
}
pub struct Health {
    pub state: State,
    pub valid: bool,
    pub timestamp_s: f64,
    pub measurement_age_s: Option<f64>,
    pub rejection: Rejection,
}
pub struct Navigator {
    pub filter: Option<Filter>,
    policy: Policy,
    now: Option<f64>,
    last_received: Option<f64>,
    last_accepted: Option<f64>,
    recovery_until: Option<f64>,
    accepted: usize,
    reason: Rejection,
    integrity_hold: Option<Rejection>,
    integrity_until: Option<f64>,
}
impl Default for Navigator {
    fn default() -> Self {
        Self::new(Policy::default())
    }
}
impl Navigator {
    pub fn new(policy: Policy) -> Self {
        assert!(policy.tracking_age_s.is_finite() && policy.tracking_age_s > 0.0);
        assert!(policy.valid_coast_s.is_finite() && policy.valid_coast_s >= policy.tracking_age_s);
        assert!(policy.lost_after_s.is_finite() && policy.lost_after_s >= policy.valid_coast_s);
        assert!(policy.max_measurement_age_s.is_finite() && policy.max_measurement_age_s >= 0.0);
        assert!(policy.max_position_sigma_m.is_finite() && policy.max_position_sigma_m > 0.0);
        Self {
            filter: None,
            policy,
            now: None,
            last_received: None,
            last_accepted: None,
            recovery_until: None,
            accepted: 0,
            reason: Rejection::None,
            integrity_hold: None,
            integrity_until: None,
        }
    }
    pub fn predict_to(&mut self, timestamp: f64) -> Result<(), Rejection> {
        if !timestamp.is_finite() || timestamp < 0.0 {
            return Err(Rejection::InvalidTime);
        }
        if let Some(now) = self.now {
            if timestamp <= now {
                return Err(Rejection::OutOfOrder);
            }
            if let Some(filter) = &mut self.filter {
                if !filter.predict(timestamp - now) {
                    self.reason = Rejection::Numerical;
                    return Err(self.reason);
                }
            }
        }
        self.now = Some(timestamp);
        Ok(())
    }
    /// Reject lateness beyond the policy tolerance. No historical rewind/replay is performed.
    pub fn observe(&mut self, captured: f64, pose: Option<&Pose>, missing: Rejection) -> bool {
        self.observe_with_covariance(
            captured,
            pose,
            &crate::default_measurement_covariance(),
            missing,
        )
    }
    /// Pixel pipelines should supply full geometry-dependent pose covariance.
    pub fn observe_with_covariance(
        &mut self,
        captured: f64,
        pose: Option<&Pose>,
        covariance: &crate::PoseCovariance,
        missing: Rejection,
    ) -> bool {
        let Some(now) = self.now else {
            self.reason = Rejection::InvalidTime;
            return false;
        };
        if !captured.is_finite() || captured < 0.0 || captured > now + 1e-9 {
            self.reason = Rejection::InvalidTime;
            return false;
        }
        if self.last_received.is_some_and(|last| captured <= last) {
            self.reason = Rejection::OutOfOrder;
            return false;
        }
        self.last_received = Some(captured);
        if now - captured > self.policy.max_measurement_age_s + 1e-9 {
            self.reason = Rejection::Stale;
            return false;
        }
        let Some(pose) = pose else {
            self.reason = missing;
            if matches!(
                missing,
                Rejection::Boundary | Rejection::Reprojection | Rejection::Ambiguous
            ) {
                self.integrity_hold = Some(missing);
            }
            return false;
        };
        if !pose.position.iter().all(|v| v.is_finite())
            || !pose.attitude.coords.iter().all(|v| v.is_finite())
        {
            self.reason = Rejection::Numerical;
            return false;
        }
        let integrity_return = self.integrity_hold.is_some();
        if !covariance.iter().all(|v| v.is_finite())
            || (covariance - covariance.transpose()).amax() > 1e-10
            || covariance.cholesky().is_none()
        {
            self.reason = Rejection::Covariance;
            return false;
        }
        let recovering = self
            .last_accepted
            .is_some_and(|t| now - t > self.policy.valid_coast_s);
        let recovery_until = if recovering {
            Some(now + 0.5)
        } else {
            self.recovery_until
        };
        let scale = recovery_until
            .map(|end| 1.0 + 3.0 * ((end - now) / 0.5).clamp(0.0, 1.0))
            .unwrap_or(1.0);
        if let Some(filter) = &mut self.filter {
            if !filter.update_with_covariance(pose, covariance, scale) {
                self.reason = Rejection::Innovation;
                return false;
            }
        } else {
            let Some(filter) = Filter::new_with_covariance(pose.clone(), covariance) else {
                self.reason = Rejection::Numerical;
                return false;
            };
            self.filter = Some(filter);
        }
        self.last_accepted = Some(captured);
        if integrity_return {
            self.integrity_until = Some(now + 0.5);
        }
        self.integrity_hold = None;
        self.recovery_until = recovery_until;
        self.accepted = if recovering || integrity_return {
            1
        } else {
            self.accepted.saturating_add(1)
        };
        self.reason = Rejection::None;
        true
    }
    pub fn health(&self) -> Health {
        let now = self.now.unwrap_or(0.0);
        let age = self.last_accepted.map(|t| now - t);
        let bounded = self.filter.as_ref().is_some_and(|f| {
            (0..3).all(|i| {
                f.covariance[(i, i)].is_finite()
                    && f.covariance[(i, i)] >= 0.0
                    && f.covariance[(i, i)]
                        <= self.policy.max_position_sigma_m * self.policy.max_position_sigma_m
            })
        });
        let state = match age {
            None => State::Acquiring,
            Some(a)
                if self.integrity_hold.is_some()
                    || self.reason == Rejection::Numerical
                    || a > self.policy.lost_after_s
                    || !bounded =>
            {
                State::Lost
            }
            Some(_) if self.accepted < 3 => State::Acquiring,
            Some(_) if self.integrity_until.is_some_and(|end| now < end) => State::Acquiring,
            Some(_) if self.recovery_until.is_some_and(|end| now < end) => State::Acquiring,
            Some(a) if a > self.policy.tracking_age_s => State::Coasting,
            Some(_) => State::Tracking,
        };
        Health {
            state,
            valid: matches!(state, State::Tracking | State::Coasting)
                && bounded
                && age.is_some_and(|a| a <= self.policy.valid_coast_s),
            timestamp_s: now,
            measurement_age_s: age,
            rejection: if self.reason != Rejection::Numerical {
                self.integrity_hold.unwrap_or(self.reason)
            } else {
                self.reason
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Q, V3};
    fn pose() -> Pose {
        Pose {
            position: V3::new(0.0, 0.0, 1.6),
            attitude: Q::identity(),
        }
    }
    #[test]
    fn visual_failures_latch_their_reason_before_any_large_r_update() {
        for failure in [Rejection::Reprojection, Rejection::Ambiguous] {
            let mut nav = Navigator::default();
            for i in 0..3 {
                let t = i as f64 * 0.1;
                nav.predict_to(t).unwrap();
                assert!(nav.observe(t, Some(&pose()), Rejection::None));
            }
            nav.predict_to(0.3).unwrap();
            let before = nav.filter.as_ref().unwrap().clone();
            let huge = crate::PoseCovariance::identity() * 1e12;
            assert!(!nav.observe_with_covariance(0.3, None, &huge, failure));
            assert_eq!(nav.filter.as_ref().unwrap().covariance, before.covariance);
            assert!(!nav.health().valid);
            assert_eq!(nav.health().state, State::Lost);
            nav.predict_to(0.4).unwrap();
            nav.observe(0.4, None, Rejection::Features);
            assert_eq!(nav.health().rejection, failure);
            nav.predict_to(0.5).unwrap();
            assert!(nav.observe(0.5, Some(&pose()), Rejection::None));
            assert!(!nav.health().valid);
            for t in [0.6, 0.7, 1.01] {
                nav.predict_to(t).unwrap();
                assert!(nav.observe(t, Some(&pose()), Rejection::None));
            }
            assert!(nav.health().valid);
        }
    }
    #[test]
    fn malformed_measurement_covariance_cannot_initialize_or_update_navigation() {
        let mut nav = Navigator::default();
        let mut covariance = crate::default_measurement_covariance();
        covariance[(0, 0)] = -1.0;
        nav.predict_to(0.0).unwrap();
        assert!(!nav.observe_with_covariance(0.0, Some(&pose()), &covariance, Rejection::None));
        assert!(nav.filter.is_none());
        assert_eq!(nav.health().rejection, Rejection::Covariance);
        for i in 1..4 {
            let t = i as f64 * 0.1;
            nav.predict_to(t).unwrap();
            nav.observe(t, Some(&pose()), Rejection::None);
        }
        let before = nav.filter.as_ref().unwrap().pose.position;
        nav.predict_to(0.4).unwrap();
        assert!(!nav.observe_with_covariance(0.4, Some(&pose()), &covariance, Rejection::None));
        assert_eq!(nav.filter.as_ref().unwrap().pose.position, before);
        assert_eq!(nav.health().rejection, Rejection::Covariance);
        nav.predict_to(0.9).unwrap();
        assert!(!nav.health().valid);
    }
    #[test]
    fn integrity_failure_latches_and_time_alone_cannot_restore_validity() {
        let mut nav = Navigator::default();
        for i in 0..3 {
            let t = i as f64 * 0.1;
            nav.predict_to(t).unwrap();
            assert!(nav.observe(t, Some(&pose()), Rejection::None));
        }
        assert!(nav.health().valid);
        nav.predict_to(0.3).unwrap();
        assert!(!nav.observe(0.3, None, Rejection::Boundary));
        assert_eq!(nav.health().state, State::Lost);
        nav.predict_to(0.4).unwrap();
        nav.observe(0.4, None, Rejection::Association);
        assert!(!nav.health().valid);
        assert_eq!(nav.health().rejection, Rejection::Boundary);
        nav.predict_to(0.5).unwrap();
        assert!(nav.observe(0.5, Some(&pose()), Rejection::None));
        nav.predict_to(1.0).unwrap();
        // A single supported pose followed by prediction is not reacquisition.
        assert!(!nav.health().valid);
        assert!(nav.observe(1.0, Some(&pose()), Rejection::None));
        assert!(!nav.health().valid);
        nav.predict_to(1.1).unwrap();
        assert!(nav.observe(1.1, Some(&pose()), Rejection::None));
        assert!(nav.health().valid);
    }
    #[test]
    fn blackout_coasts_invalidates_then_recovers_without_reset() {
        let mut nav = Navigator::default();
        assert_eq!(nav.health().state, State::Acquiring);
        for i in 0..20 {
            let t = i as f64 * 0.1;
            nav.predict_to(t).unwrap();
            assert!(nav.observe(t, Some(&pose()), Rejection::None));
        }
        let before = nav.filter.as_ref().unwrap().covariance.trace();
        nav.predict_to(2.2).unwrap();
        assert_eq!(nav.health().state, State::Coasting);
        assert!(nav.health().valid);
        nav.predict_to(4.9).unwrap();
        assert_eq!(nav.health().state, State::Coasting);
        assert!(!nav.health().valid);
        assert!(nav.filter.as_ref().unwrap().covariance.trace() > before);
        assert!(nav.observe(4.9, Some(&pose()), Rejection::None));
        assert_eq!(nav.health().state, State::Acquiring);
        assert!(!nav.health().valid);
        nav.predict_to(5.5).unwrap();
        assert!(nav.observe(5.5, Some(&pose()), Rejection::None));
        // A fresh gap beyond the valid coasting limit starts another recovery window.
        nav.predict_to(5.6).unwrap();
        assert!(nav.observe(5.6, Some(&pose()), Rejection::None));
        for i in 0..6 {
            let t = 5.7 + i as f64 * 0.1;
            nav.predict_to(t).unwrap();
            nav.observe(t, Some(&pose()), Rejection::None);
        }
        assert_eq!(nav.health().state, State::Tracking);
        nav.predict_to(10.0).unwrap();
        assert_eq!(nav.health().state, State::Lost);
    }
    #[test]
    fn rejects_stale_duplicate_future_and_nonfinite_timestamps() {
        let mut nav = Navigator::default();
        nav.predict_to(1.0).unwrap();
        assert!(!nav.observe(0.5, Some(&pose()), Rejection::None));
        assert_eq!(nav.health().rejection, Rejection::Stale);
        assert!(!nav.observe(0.5, Some(&pose()), Rejection::None));
        assert_eq!(nav.health().rejection, Rejection::OutOfOrder);
        assert!(!nav.observe(2.0, Some(&pose()), Rejection::None));
        assert!(nav.predict_to(f64::NAN).is_err());
        assert!(nav.predict_to(0.9).is_err());
        assert_eq!(nav.health().timestamp_s, 1.0);
        assert!(nav.filter.is_none());
    }
}
