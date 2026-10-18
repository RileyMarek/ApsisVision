//! Host-side integration example; the library itself remains no_std.
use apsis_core::{
    math::{Pose, Q, V3},
    navigation::{Navigator, Rejection, State},
};
fn main() {
    let mut nav = Navigator::default();
    let mut previous = None;
    for tick in 0..400 {
        let time = tick as f64 * 0.02;
        nav.predict_to(time).unwrap();
        if tick % 5 == 0 {
            let missing = (100..250).contains(&tick);
            // Replace this example observation with a calibrated image pose and capture timestamp.
            let measured = Pose {
                position: V3::new(0.0, 0.0, 1.6),
                attitude: Q::identity(),
            };
            nav.observe(
                time,
                if missing { None } else { Some(&measured) },
                Rejection::NoImage,
            );
        }
        let health = nav.health();
        if previous != Some((health.state, health.valid)) {
            println!(
                "t={time:.2}s state={} GNC-valid={} measurement-age={:?} reason={}",
                health.state.label(),
                health.valid,
                health.measurement_age_s,
                health.rejection.label()
            );
            previous = Some((health.state, health.valid));
        }
        if health.valid {
            std::hint::black_box(nav.filter.as_ref());
        } // Permit consumer use only when valid.
        if tick == 200 {
            assert_eq!(health.state, State::Coasting);
            assert!(!health.valid);
        }
    }
    assert!(nav.health().valid);
}
