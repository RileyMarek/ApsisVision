//! Software-in-loop rendezvous reference; pose-sensor and pixel modes are reported separately.
use apsis_core::{
    guidance::{Guidance, Input, Mode},
    math::{landmarks, target, Camera, Pose, Q, V3},
    navigation::{Navigator, Rejection},
    vision::{corners, integrity, solve_admitted},
};
use apsis_sim::{
    dynamics::{contact_acceleration, RelativeState, RigidBody},
    scene, Noise,
};
use std::{error::Error, fs, path::PathBuf};

fn run(
    out: PathBuf,
    pixels: bool,
    tumble: bool,
    outage: Option<f64>,
) -> Result<serde_json::Value, Box<dyn Error>> {
    fs::create_dir_all(&out)?;
    let dt = 0.02;
    let n = 0.0011;
    let camera = Camera::new(640, 480, 320.0);
    let to_camera = Q::from_scaled_axis(V3::new(core::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let mut relative = RelativeState {
        position: V3::new(0.0, -2.0, 0.0),
        velocity: V3::zeros(),
    };
    let initial = to_camera.inverse();
    let mut body = RigidBody {
        attitude: initial,
        rate_body: initial.inverse() * V3::new(0.0, 0.0, n)
            + if tumble {
                V3::new(0.04, 0.05, 0.1)
            } else {
                V3::zeros()
            },
        inertia: V3::new(0.002, 0.0004, 0.002),
    };
    let mut nav = Navigator::default();
    let mut guidance = Guidance::default();
    let mut rng = Noise::new(42);
    let fixed_prior = Pose {
        position: V3::new(0.0, 0.0, 2.0),
        attitude: Q::identity(),
    };
    let mut csv=String::from("time_s,mode,reason,nav_state,valid,range_m,true_range_m,px_m,py_m,pz_m,ax_m_s2,ay_m_s2,az_m_s2,position_error_m,contact_acceleration_m_s2\n");
    let mut events = Vec::new();
    let mut previous = None;
    let mut captured = false;
    let mut worst = 0.0_f64;
    let mut valid_ticks = 0;
    let mut rejected = 0;
    let mut accepted = 0;
    for tick in 0..9000 {
        let t = tick as f64 * dt;
        let truth = Pose {
            position: to_camera * (-relative.position),
            attitude: to_camera * Q::from_scaled_axis(V3::new(0.0, 0.0, -n * t)) * body.attitude,
        };
        nav.predict_to(t)
            .map_err(|r| format!("predict: {}", r.label()))?;
        if tick % 5 == 0 {
            let unavailable = outage.is_some_and(|start| (start..start + 3.0).contains(&t));
            let mut covariance = apsis_core::default_measurement_covariance();
            let mut missing = Rejection::PoseFit;
            let measurement = if unavailable {
                None
            } else if pixels {
                let image =
                    scene::render(camera, &truth, t, scene::Conditions::default(), &mut rng);
                let features = corners::detect(&image.gray, camera);
                let prior = nav.filter.as_ref().map(|f| &f.pose).unwrap_or(&fixed_prior);
                let matches = corners::associate(camera, &landmarks(), &features, prior, 20.0);
                solve_admitted(camera, &landmarks(), &matches, prior, 1.0)
                    .map_err(|reason| {
                        missing = reason.into();
                    })
                    .ok()
                    .and_then(|(p, _, uncertainty)| {
                        covariance += uncertainty.covariance;
                        integrity::cubesat_modes(&image.gray, camera, &p)
                            .map_err(|reason| {
                                missing = reason.into();
                            })
                            .ok()
                            .map(|_| p)
                    })
            } else {
                // A declared stochastic pose sensor, not camera-derived evidence.
                let mut p = truth.clone();
                p.position += V3::new(rng.uniform(), rng.uniform(), rng.uniform()) * 0.001732;
                p.attitude = Q::from_scaled_axis(
                    V3::new(rng.uniform(), rng.uniform(), rng.uniform()) * 0.0005,
                ) * p.attitude;
                Some(p)
            };
            if nav.observe_with_covariance(
                t,
                measurement.as_ref(),
                &covariance,
                if unavailable {
                    Rejection::NoImage
                } else {
                    missing
                },
            ) {
                accepted += 1;
            } else {
                rejected += 1;
            }
        }
        let h = nav.health();
        let f = nav.filter.as_ref();
        let estimate = f.map(|f| &f.pose).unwrap_or(&fixed_prior);
        let v = f.map(|f| f.velocity).unwrap_or(V3::zeros());
        let w = f.map(|f| f.rate).unwrap_or(V3::zeros());
        let sigma = f
            .map(|f| {
                (0..3)
                    .map(|i| f.covariance[(i, i)].sqrt())
                    .fold(0.0_f64, f64::max)
            })
            .unwrap_or(f64::INFINITY);
        let in_fov = target().iter().all(|p| {
            camera
                .project(estimate.transform(*p))
                .is_some_and(|p| p.x >= 5.0 && p.x < 635.0 && p.y >= 5.0 && p.y < 475.0)
        });
        let command = guidance.update(Input {
            pose: estimate,
            velocity: v,
            angular_rate: w,
            valid: h.valid,
            timestamp_s: t,
            sigma_position_m: sigma,
            in_fov,
        });
        let error = (estimate.position - truth.position).norm();
        if h.valid {
            valid_ticks += 1;
            worst = worst.max(error);
        }
        if previous != Some(command.mode) {
            events.push(serde_json::json!({"t":t,"mode":command.mode.label(),"reason":command.reason,"true_range_m":truth.position.z}));
            previous = Some(command.mode);
        }
        let contact = contact_acceleration(truth.position.z, (to_camera * (-relative.velocity)).z);
        csv.push_str(&format!(
            "{t},{},{},{},{},{},{},{},{},{},{},{},{},{error},{contact}\n",
            command.mode.label(),
            command.reason,
            h.state.label(),
            h.valid,
            estimate.position.z,
            truth.position.z,
            truth.position.x,
            truth.position.y,
            truth.position.z,
            command.relative_acceleration.x,
            command.relative_acceleration.y,
            command.relative_acceleration.z
        ));
        if command.mode == Mode::Capture {
            captured = true;
            break;
        }
        let relative_camera_acceleration =
            command.relative_acceleration + V3::new(0.0, 0.0, contact);
        relative.step(dt, n, -(to_camera.inverse() * relative_camera_acceleration));
        body.step(dt);
        if !relative
            .position
            .iter()
            .chain(relative.velocity.iter())
            .all(|x| x.is_finite())
        {
            return Err("nonfinite plant".into());
        }
    }
    fs::write(out.join("trajectory.csv"), csv)?;
    let report = serde_json::json!({"sensor":if pixels{"pixel_pipeline"}else{"pose_sensor_reference"},"not_vision_validation":!pixels,"tumble":tumble,"outage":outage,"captured":captured,"accepted":accepted,"rejected":rejected,"valid_ticks":valid_ticks,"worst_valid_position_m":if valid_ticks>0{Some(worst)}else{None},"events":events,"assumptions":{"reference_orbit_mean_motion_rad_s":n,"camera_fx_px":320,"camera_to_chaser_capture_plane_m":0.30,"target_center_to_port_m":0.05,"contact":"axial Kelvin-Voigt unit model only; mission stops at virtual capture before contact; no physical latch or hardware"}});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let (mut pixels, mut tumble, mut outage) = (false, false, None);
    let mut out = PathBuf::from("output/mission");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pixels" => pixels = true,
            "--tumble" => tumble = true,
            "--outage" => outage = Some(40.0),
            "--outage-start" => {
                let t: f64 = args.next().ok_or("missing outage time")?.parse()?;
                if !t.is_finite() || !(0.0..177.0).contains(&t) {
                    return Err("outage start must be 0..177 seconds".into());
                }
                outage = Some(t);
            }
            "--out" => out = PathBuf::from(args.next().ok_or("missing output directory")?),
            _ => return Err(format!("unknown mission option: {arg}").into()),
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&run(out, pixels, tumble, outage)?)?
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_sensor_closes_loop_and_capture_is_explicitly_scoped() {
        let r = run(
            std::env::temp_dir().join(format!("apsis-mission-{}", std::process::id())),
            false,
            false,
            None,
        )
        .unwrap();
        assert!(r["captured"].as_bool().unwrap(), "{r}");
        assert_eq!(r["sensor"], "pose_sensor_reference");
        assert!(r["not_vision_validation"].as_bool().unwrap());
    }
    #[test]
    fn navigation_loss_near_port_aborts_instead_of_capturing() {
        let r = run(
            std::env::temp_dir().join(format!("apsis-abort-{}", std::process::id())),
            false,
            false,
            Some(75.0),
        )
        .unwrap();
        assert_eq!(r["captured"], false);
        assert!(r["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["mode"] == "abort"));
    }
}
