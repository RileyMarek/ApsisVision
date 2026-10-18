use apsis_core::math::{landmarks, target, Camera, Pose, Q, V3};
use apsis_core::navigation::{Navigator, Rejection};
use apsis_core::vision::{corners, solve_uncertain};
use apsis_sim::scene::{self, Conditions};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    error::Error,
    fs,
    io::BufWriter,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Serialize, Deserialize)]
struct Config {
    range_m: f64,
    tumble_scale: f64,
    phase_rad: f64,
    light_gain: f64,
    sun_angle_rad: f64,
    save_images: bool,
    frames: usize,
    seed: u64,
    noise: f64,
    occlusion: f64,
    false_detections: usize,
    calibration_bias: f64,
    delay_ms: f64,
    blackout_start: f64,
    blackout_duration: f64,
    glare: f64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            range_m: 1.6,
            tumble_scale: 1.0,
            phase_rad: 0.0,
            light_gain: 1.0,
            sun_angle_rad: 0.0,
            save_images: true,
            frames: 1000,
            seed: 42,
            noise: 1.0,
            occlusion: 0.0,
            false_detections: 0,
            calibration_bias: 0.0,
            delay_ms: 0.0,
            blackout_start: 8.0,
            blackout_duration: 3.0,
            glare: 0.8,
        }
    }
}
impl Config {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        for (name, value, lo, hi) in [
            ("range", self.range_m, 0.5, 20.0),
            ("tumble scale", self.tumble_scale, 0.0, 5.0),
            (
                "initial phase",
                self.phase_rad,
                -core::f64::consts::PI,
                core::f64::consts::PI,
            ),
            ("light gain", self.light_gain, 0.05, 5.0),
            (
                "sun angle",
                self.sun_angle_rad,
                -core::f64::consts::PI,
                core::f64::consts::PI,
            ),
            ("noise", self.noise, 0.0, 25.0),
            ("occlusion", self.occlusion, 0.0, 0.9),
            ("calibration bias", self.calibration_bias, -0.15, 0.15),
            ("delay", self.delay_ms, 0.0, 2000.0),
            ("blackout start", self.blackout_start, 0.0, 60.0),
            ("blackout duration", self.blackout_duration, 0.0, 8.0),
            ("glare", self.glare, 0.0, 3.0),
        ] {
            if !value.is_finite() || !(lo..=hi).contains(&value) {
                return Err(format!("{name} must be in [{lo}, {hi}]").into());
            }
        }
        if !(10..=3000).contains(&self.frames) || self.false_detections > 40 {
            return Err("frames must be 10..3000 and false detections 0..40".into());
        }
        Ok(())
    }
}
#[derive(Serialize)]
struct Tick {
    t: f64,
    captured: Option<f64>,
    state: &'static str,
    valid: bool,
    age: Option<f64>,
    reason: &'static str,
    used: bool,
    available: bool,
    scene: String,
    features: Vec<[f64; 2]>,
    matches: Vec<[f64; 3]>,
    wire: Vec<[f64; 2]>,
    plate: Vec<[f64; 2]>,
    p: Option<[f64; 3]>,
    v: Option<[f64; 3]>,
    w: Option<[f64; 3]>,
    q: Option<[f64; 4]>,
    sigma: Option<[f64; 12]>,
    truth_p: [f64; 3],
    truth_q: [f64; 4],
    error: Option<f64>,
    angle: Option<f64>,
    nees_12: Option<f64>,
    last_nis_6: Option<f64>,
    measurement_sigma: Option<[f64; 6]>,
    geometry_condition: Option<f64>,
    fit_rms: Option<f64>,
    vision_us: u128,
    filter_us: u128,
    correction_m: f64,
    correction_deg: f64,
}
#[derive(Serialize)]
struct Summary {
    position_rms_m: Option<f64>,
    attitude_rms_deg: Option<f64>,
    worst_position_m: Option<f64>,
    worst_attitude_deg: Option<f64>,
    blackout_worst_position_m: Option<f64>,
    blackout_worst_attitude_deg: Option<f64>,
    recovery_s: Option<f64>,
    optical_return_s: Option<f64>,
    reacquisition_jump_m: Option<f64>,
    reacquisition_jump_deg: Option<f64>,
    accepted: usize,
    camera_frames: usize,
    valid_fraction: f64,
    vision_p50_us: u128,
    vision_p95_us: u128,
    vision_p99_us: u128,
    filter_p95_us: u128,
    filter_bytes: usize,
    navigator_bytes: usize,
}
#[derive(Serialize)]
struct Recording {
    scenarios: serde_json::Value,
    config: Config,
    summary: Summary,
    frames: Vec<Tick>,
}
struct Packet {
    captured: f64,
    release: f64,
    covariance: apsis_core::PoseCovariance,
    pose: Option<Pose>,
    reason: Rejection,
}
fn xyz(p: V3) -> [f64; 3] {
    [p.x, p.y, p.z]
}
fn quat(q: Q) -> [f64; 4] {
    let q = q.quaternion();
    [q.w, q.i, q.j, q.k]
}
fn png_file(path: &Path, camera: Camera, rgb: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut encoder = png::Encoder::new(
        BufWriter::new(fs::File::create(path)?),
        camera.width as u32,
        camera.height as u32,
    );
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    encoder.write_header()?.write_image_data(rgb)?;
    Ok(())
}
fn percentile(values: &mut [u128], fraction: f64) -> u128 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[((values.len() - 1) as f64 * fraction).round() as usize]
}
fn run(config: Config, out: PathBuf) -> Result<Recording, Box<dyn Error>> {
    config.validate()?;
    fs::create_dir_all(out.join("frames"))?;
    let camera = Camera::new(640, 480, 1500.0);
    let mut calibrated = camera;
    calibrated.fx *= 1.0 + config.calibration_bias;
    calibrated.fy *= 1.0 + config.calibration_bias;
    let mut noise = apsis_sim::Noise::new(config.seed);
    let mut nav = Navigator::default();
    let seed = Pose {
        position: V3::new(0.0, 0.0, 1.6),
        attitude: Q::from_scaled_axis(V3::new(0.2, 0.3, 0.1)),
    };
    let mut packets = VecDeque::new();
    let mut frames = Vec::with_capacity(config.frames);
    let mut last_scene = String::new();
    let mut captured = None;
    let mut available = false;
    let mut features = Vec::new();
    let mut matched = Vec::new();
    let mut fit_rms = None;
    let mut measurement_sigma = None;
    let mut geometry_condition = None;
    let (mut sum, mut angles, mut scored, mut accepted, mut valid_count, mut camera_frames) =
        (0.0, 0.0, 0, 0, 0, 0);
    let (mut worst, mut worst_angle, mut blackout_worst, mut blackout_angle) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut recovery, mut jump, mut jump_angle) = (None, None, None);
    let mut optical_return = None;
    let mut blackout_scored = 0;
    let end = config.blackout_start + config.blackout_duration;
    let mut vision_times = Vec::new();
    let mut filter_times = Vec::new();
    for frame in 0..config.frames {
        let t = frame as f64 * 0.02;
        let mut truth = apsis_sim::truth(t * config.tumble_scale);
        // Independent scenario truth. The estimator's fixed prior is unchanged.
        truth.position = V3::new(
            0.012 * (t * 0.2).sin(),
            0.008 * (t * 0.3).sin(),
            config.range_m - t * 0.0005,
        );
        truth.attitude = Q::from_scaled_axis(V3::new(0.0, 0.0, config.phase_rad)) * truth.attitude;
        let filter_start = Instant::now();
        nav.predict_to(t)
            .map_err(|r| format!("predict failed: {}", r.label()))?;
        let mut filter_us = filter_start.elapsed().as_micros();
        let mut vision_us = 0;
        if frame % 5 == 0 {
            camera_frames += 1;
            captured = Some(t);
            available =
                !(config.blackout_duration > 0.0 && t >= config.blackout_start && t < end - 1e-9);
            let image = scene::render(
                camera,
                &truth,
                t,
                Conditions {
                    light_gain: config.light_gain,
                    sun_angle_rad: config.sun_angle_rad,
                    noise: config.noise,
                    occlusion: config.occlusion,
                    false_blobs: config.false_detections,
                    glare: config.glare,
                },
                &mut noise,
            );
            last_scene = format!("frames/{frame:06}.png");
            if config.save_images {
                png_file(&out.join(&last_scene), camera, &image.rgb)?;
            }
            features.clear();
            matched.clear();
            fit_rms = None;
            measurement_sigma = None;
            geometry_condition = None;
            let start = Instant::now();
            let prior = nav.filter.as_ref().map(|f| &f.pose).unwrap_or(&seed);
            let mut measurement_covariance = apsis_core::default_measurement_covariance();
            let (pose, reason) = if available {
                let detected = corners::detect(&image.gray, calibrated);
                features = detected.iter().map(|f| [f.pixel.x, f.pixel.y]).collect();
                let visible: [bool; 12] = std::array::from_fn(|i| {
                    let point = prior.transform(landmarks()[i]);
                    if i >= 8 {
                        return (prior.attitude * V3::new(0.0, 0.0, -1.0)).dot(&point) < 0.0;
                    }
                    (0..3).any(|axis| {
                        let mut normal = V3::zeros();
                        normal[axis] = if i & (1 << axis) == 0 { -1.0 } else { 1.0 };
                        (prior.attitude * normal).dot(&point) < 0.0
                    })
                });
                let matches = corners::associate_visible(
                    calibrated,
                    &landmarks(),
                    &detected,
                    prior,
                    if nav.filter.is_some() { 12.0 } else { 20.0 },
                    &visible,
                );
                matched = matches
                    .iter()
                    .map(|(id, p)| [*id as f64, p.x, p.y])
                    .collect();
                if detected.len() < 6 {
                    (None, Rejection::Features)
                } else if matches.len() < 6 {
                    (None, Rejection::Association)
                } else if let Some((p, rms, uncertainty)) =
                    solve_uncertain(calibrated, &landmarks(), &matches, prior, 1.0)
                {
                    fit_rms = Some(rms);
                    measurement_covariance += uncertainty.covariance;
                    measurement_sigma = Some(std::array::from_fn(|i| {
                        measurement_covariance[(i, i)].sqrt()
                    }));
                    geometry_condition = Some(uncertainty.condition);
                    matched.retain(|m| {
                        calibrated
                            .project(p.transform(landmarks()[m[0] as usize]))
                            .is_some_and(|pixel| {
                                (pixel - apsis_core::math::Vector2::new(m[1], m[2])).norm() < 2.5
                            })
                    });
                    if apsis_core::vision::silhouette::verify(
                        &image.gray,
                        calibrated,
                        &target(),
                        &p,
                    )
                    .is_some()
                    {
                        (Some(p), Rejection::None)
                    } else {
                        (None, Rejection::Boundary)
                    }
                } else {
                    (None, Rejection::PoseFit)
                }
            } else {
                (None, Rejection::NoImage)
            };
            vision_us = start.elapsed().as_micros();
            vision_times.push(vision_us);
            packets.push_back(Packet {
                captured: t,
                release: t + config.delay_ms / 1000.0,
                pose,
                covariance: measurement_covariance,
                reason,
            });
        }
        let before = nav.filter.as_ref().map(|f| f.pose.clone());
        let mut used = false;
        let update_start = Instant::now();
        while packets.front().is_some_and(|p| p.release <= t + 1e-9) {
            let packet = packets.pop_front().unwrap();
            used |= nav.observe_with_covariance(
                packet.captured,
                packet.pose.as_ref(),
                &packet.covariance,
                packet.reason,
            );
        }
        filter_us += update_start.elapsed().as_micros();
        filter_times.push(filter_us);
        if used {
            accepted += 1;
        }
        let health = nav.health();
        if health.valid {
            valid_count += 1;
        }
        let mut tick = Tick {
            t,
            captured,
            state: health.state.label(),
            valid: health.valid,
            age: health.measurement_age_s,
            reason: health.rejection.label(),
            used,
            available,
            scene: last_scene.clone(),
            features: features.clone(),
            matches: matched.clone(),
            wire: Vec::new(),
            plate: Vec::new(),
            p: None,
            v: None,
            w: None,
            q: None,
            sigma: None,
            truth_p: xyz(truth.position),
            truth_q: quat(truth.attitude),
            error: None,
            angle: None,
            nees_12: None,
            last_nis_6: nav.filter.as_ref().and_then(|f| f.last_nis),
            measurement_sigma,
            geometry_condition,
            fit_rms,
            vision_us,
            filter_us,
            correction_m: 0.0,
            correction_deg: 0.0,
        };
        if let Some(f) = &nav.filter {
            let error = (f.pose.position - truth.position).norm();
            let angle = (f.pose.attitude * truth.attitude.inverse())
                .angle()
                .to_degrees();
            tick.p = Some(xyz(f.pose.position));
            tick.v = Some(xyz(f.velocity));
            tick.w = Some(xyz(f.rate));
            tick.q = Some(quat(f.pose.attitude));
            tick.sigma = Some(std::array::from_fn(|i| f.covariance[(i, i)].sqrt()));
            tick.error = Some(error);
            tick.angle = Some(angle);
            // Consistency scoring is downstream of estimation, never an update input.
            let mut state_error = apsis_core::math::SVector::<f64, 12>::zeros();
            state_error
                .fixed_rows_mut::<3>(0)
                .copy_from(&(f.pose.position - truth.position));
            let truth_velocity =
                V3::new(0.0024 * (t * 0.2).cos(), 0.0024 * (t * 0.3).cos(), -0.0005);
            state_error
                .fixed_rows_mut::<3>(3)
                .copy_from(&(f.velocity - truth_velocity));
            state_error
                .fixed_rows_mut::<3>(6)
                .copy_from(&(f.pose.attitude * truth.attitude.inverse()).scaled_axis());
            let truth_rate = Q::from_scaled_axis(V3::new(0.0, 0.0, config.phase_rad))
                * V3::new(0.035, 0.05, 0.14)
                * config.tumble_scale;
            state_error
                .fixed_rows_mut::<3>(9)
                .copy_from(&(f.rate - truth_rate));
            tick.nees_12 = f
                .covariance
                .cholesky()
                .map(|c| state_error.dot(&c.solve(&state_error)));
            tick.wire = target()
                .iter()
                .filter_map(|p| calibrated.project(f.pose.transform(*p)))
                .map(|p| [p.x, p.y])
                .collect();
            tick.plate = [8, 9, 11, 10]
                .iter()
                .filter_map(|i| calibrated.project(f.pose.transform(landmarks()[*i])))
                .map(|p| [p.x, p.y])
                .collect();
            if let Some(before) = before {
                tick.correction_m = (f.pose.position - before.position).norm();
                tick.correction_deg = (f.pose.attitude * before.attitude.inverse())
                    .angle()
                    .to_degrees();
            }
            sum += error * error;
            angles += angle * angle;
            scored += 1;
            worst = worst.max(error);
            worst_angle = worst_angle.max(angle);
            if t >= config.blackout_start && t < end {
                blackout_scored += 1;
                blackout_worst = blackout_worst.max(error);
                blackout_angle = blackout_angle.max(angle);
            }
            if config.blackout_duration > 0.0 && t >= end && used && optical_return.is_none() {
                optical_return = Some(t - end);
                jump = Some(tick.correction_m);
                jump_angle = Some(tick.correction_deg);
            }
            if config.blackout_duration > 0.0 && t >= end && health.valid && recovery.is_none() {
                recovery = Some(t - end);
            }
        }
        frames.push(tick);
    }
    let summary = Summary {
        position_rms_m: (scored > 0).then(|| (sum / scored as f64).sqrt()),
        attitude_rms_deg: (scored > 0).then(|| (angles / scored as f64).sqrt()),
        worst_position_m: (scored > 0).then_some(worst),
        worst_attitude_deg: (scored > 0).then_some(worst_angle),
        blackout_worst_position_m: (blackout_scored > 0).then_some(blackout_worst),
        blackout_worst_attitude_deg: (blackout_scored > 0).then_some(blackout_angle),
        recovery_s: recovery,
        optical_return_s: optical_return,
        reacquisition_jump_m: jump,
        reacquisition_jump_deg: jump_angle,
        accepted,
        camera_frames,
        valid_fraction: valid_count as f64 / config.frames as f64,
        vision_p50_us: percentile(&mut vision_times, 0.5),
        vision_p95_us: percentile(&mut vision_times, 0.95),
        vision_p99_us: percentile(&mut vision_times, 0.99),
        filter_p95_us: percentile(&mut filter_times, 0.95),
        filter_bytes: std::mem::size_of::<apsis_core::Filter>(),
        navigator_bytes: std::mem::size_of::<Navigator>(),
    };
    let recording = Recording {
        scenarios: serde_json::from_str(include_str!("../../../scenarios/demo.json"))?,
        config,
        summary,
        frames,
    };
    let json = serde_json::to_string(&recording)?;
    fs::write(out.join("data.json"), &json)?;
    if recording.config.save_images {
        fs::write(
            out.join("index.html"),
            include_str!("viewer.html")
                .replace("__STYLE__", include_str!("viewer.css"))
                .replace("__SCRIPT__", include_str!("viewer.js"))
                .replace("__RECORDING__", &json),
        )?;
    }
    let mut csv=String::from("time_s,capture_s,state,valid,measurement_age_s,rejection,accepted,camera_available,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,wx_rad_s,wy_rad_s,wz_rad_s,position_error_m,attitude_error_deg,vision_us,filter_us,sigma_px_m,sigma_py_m,sigma_pz_m,sigma_vx_m_s,sigma_vy_m_s,sigma_vz_m_s,sigma_theta_x_rad,sigma_theta_y_rad,sigma_theta_z_rad,sigma_wx_rad_s,sigma_wy_rad_s,sigma_wz_rad_s,qw,qx,qy,qz,nees_12,last_nis_6,geometry_condition,measurement_sigma_x_m,measurement_sigma_y_m,measurement_sigma_z_m,measurement_sigma_rx_rad,measurement_sigma_ry_rad,measurement_sigma_rz_rad\n");
    for f in &recording.frames {
        let mut row = vec![
            f.t.to_string(),
            f.captured.map(|v| v.to_string()).unwrap_or_default(),
            f.state.into(),
            f.valid.to_string(),
            f.age.map(|v| v.to_string()).unwrap_or_default(),
            f.reason.into(),
            f.used.to_string(),
            f.available.to_string(),
        ];
        for v in [f.p, f.v, f.w] {
            row.extend((0..3).map(|i| v.map(|a| a[i].to_string()).unwrap_or_default()));
        }
        row.push(f.error.map(|v| v.to_string()).unwrap_or_default());
        row.push(f.angle.map(|v| v.to_string()).unwrap_or_default());
        row.push(f.vision_us.to_string());
        row.push(f.filter_us.to_string());
        row.extend((0..12).map(|i| f.sigma.map(|a| a[i].to_string()).unwrap_or_default()));
        row.extend((0..4).map(|i| f.q.map(|a| a[i].to_string()).unwrap_or_default()));
        row.extend(
            [f.nees_12, f.last_nis_6, f.geometry_condition]
                .map(|v| v.map(|x| x.to_string()).unwrap_or_default()),
        );
        row.extend((0..6).map(|i| {
            f.measurement_sigma
                .map(|a| a[i].to_string())
                .unwrap_or_default()
        }));
        csv.push_str(&row.join(","));
        csv.push('\n');
    }
    fs::write(out.join("telemetry.csv"), csv)?;
    Ok(recording)
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut config = Config::default();
    let mut out = PathBuf::from("output");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--no-images" {
            config.save_images = false;
            continue;
        }
        if arg == "--help" {
            println!("apsis-cli [--out output] [--no-images] [--frames 1000] [--seed 42] [--noise 1] [--occlusion 0] [--false-detections 0] [--calibration-bias 0] [--delay-ms 0] [--glare 0.8] [--blackout-start 8] [--blackout-duration 3] [--range-m 1.6] [--tumble-scale 1] [--phase-rad 0] [--light-gain 1] [--sun-angle-rad 0]");
            return Ok(());
        }
        let value = args.next().ok_or("missing argument value")?;
        match arg.as_str() {
            "--out" => out = PathBuf::from(value),
            "--frames" => config.frames = value.parse()?,
            "--seed" => config.seed = value.parse()?,
            "--range-m" => config.range_m = value.parse()?,
            "--tumble-scale" => config.tumble_scale = value.parse()?,
            "--phase-rad" => config.phase_rad = value.parse()?,
            "--light-gain" => config.light_gain = value.parse()?,
            "--sun-angle-rad" => config.sun_angle_rad = value.parse()?,
            "--noise" => config.noise = value.parse()?,
            "--occlusion" => config.occlusion = value.parse()?,
            "--false-detections" => config.false_detections = value.parse()?,
            "--calibration-bias" => config.calibration_bias = value.parse()?,
            "--delay-ms" => config.delay_ms = value.parse()?,
            "--glare" => config.glare = value.parse()?,
            "--blackout-start" => config.blackout_start = value.parse()?,
            "--blackout-duration" => config.blackout_duration = value.parse()?,
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let r = run(config, out.clone())?;
    println!("{}", serde_json::to_string_pretty(&r.summary)?);
    println!(
        "Wrote {}",
        out.join(if r.config.save_images {
            "index.html"
        } else {
            "data.json"
        })
        .display()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clutter_and_noisy_reacquisition_cannot_repeat_false_valid_regressions() {
        for (name, config) in [
            (
                "clutter",
                Config {
                    false_detections: 16,
                    save_images: false,
                    ..Config::default()
                },
            ),
            (
                "noise",
                Config {
                    noise: 12.0,
                    seed: 43,
                    save_images: false,
                    ..Config::default()
                },
            ),
        ] {
            let path =
                std::env::temp_dir().join(format!("apsis-integrity-{name}-{}", std::process::id()));
            let r = run(config, path).unwrap();
            // Weighting can prevent a bad fit reaching the boundary check at all.
            // Keep the behavior guards below: nonzero availability and bounded valid error.
            assert!(r.frames.iter().any(|f| matches!(
                f.reason,
                "unsupported_boundary" | "innovation_gate" | "pose_fit_failed"
            )));
            let valid: Vec<_> = r.frames.iter().filter(|f| f.valid).collect();
            assert!(
                !valid.is_empty(),
                "must not pass by disabling navigation entirely"
            );
            assert!(
                valid.iter().all(|f| f.error.is_some_and(|e| e < 0.15)),
                "{name}: false-valid position regression"
            );
        }
    }
    #[test]
    fn default_twenty_second_recording_has_bounded_recovery() {
        let path = std::env::temp_dir().join(format!("apsis-default-{}", std::process::id()));
        let r = run(Config::default(), path).unwrap();
        assert!(r.summary.position_rms_m.unwrap() < 0.08);
        assert!(r.summary.worst_position_m.unwrap() < 0.15);
        assert!(r.summary.attitude_rms_deg.unwrap() < 8.0);
        assert!(r.summary.recovery_s.is_some_and(|t| t <= 1.0));
        assert!(r.frames[400..550].iter().all(|f| !f.available && !f.used));
        assert!(r.frames[430..550].iter().all(|f| !f.valid));
        assert_eq!(r.frames[580].state, "tracking");
    }
    #[test]
    fn geometric_image_pipeline_coasts_three_seconds_and_recovers() {
        let path = std::env::temp_dir().join(format!("apsis-geometric-{}", std::process::id()));
        let r = run(
            Config {
                frames: 1000,
                blackout_start: 4.0,
                ..Config::default()
            },
            path,
        )
        .unwrap();
        assert!(r.summary.accepted > 40, "accepted {}", r.summary.accepted);
        assert!(r.summary.recovery_s.is_some(), "no recovery");
        assert!(
            r.summary.position_rms_m.unwrap() < 0.08,
            "position {:?}",
            r.summary.position_rms_m
        );
        assert!(
            r.summary.attitude_rms_deg.unwrap() < 12.0,
            "attitude {:?}",
            r.summary.attitude_rms_deg
        );
        let during = &r.frames[300];
        assert_eq!(during.state, "coasting");
        assert!(!during.valid);
        assert!(!during.available);
        assert!(during.sigma.unwrap()[0] > r.frames[195].sigma.unwrap()[0]);
        assert!(r
            .frames
            .iter()
            .filter_map(|f| f.p)
            .flatten()
            .all(|v| v.is_finite()));
    }
    #[test]
    fn stale_camera_never_initializes_navigation() {
        let path = std::env::temp_dir().join(format!("apsis-delay-{}", std::process::id()));
        let r = run(
            Config {
                save_images: false,
                frames: 100,
                delay_ms: 300.0,
                ..Config::default()
            },
            path.clone(),
        )
        .unwrap();
        assert_eq!(r.summary.accepted, 0);
        assert!(r.frames.iter().all(|f| f.p.is_none() && !f.valid));
        assert!(r.frames.iter().any(|f| f.reason == "stale_measurement"));
        assert!(path.join("data.json").is_file());
        assert!(path.join("telemetry.csv").is_file());
        assert!(!path.join("index.html").exists());
        assert_eq!(fs::read_dir(path.join("frames")).unwrap().count(), 0);
    }
}
