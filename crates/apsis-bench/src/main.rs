//! Benchmark rectified grayscale images. Truth is consumed only after inference.
use apsis_core::{
    math::{Camera, Pose, Q, V3},
    navigation::{Navigator, Rejection},
    vision::{corners, solve_robust},
};
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Deserialize)]
struct Calibration {
    width: usize,
    height: usize,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    #[serde(default)]
    distortion: Vec<f64>,
}
#[derive(Clone, Deserialize)]
struct PoseData {
    position_m: [f64; 3],
    quaternion_wxyz: [f64; 4],
}
impl PoseData {
    fn pose(&self) -> Result<Pose, Box<dyn Error>> {
        let q = self.quaternion_wxyz;
        if !self
            .position_m
            .iter()
            .chain(q.iter())
            .all(|v| v.is_finite())
            || (q.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() > 1e-4
        {
            return Err(
                "pose requires finite meters and a unit body-to-camera quaternion (w,x,y,z)".into(),
            );
        }
        Ok(Pose {
            position: V3::from_row_slice(&self.position_m),
            attitude: Q::from_quaternion(apsis_core::math::Quaternion::new(q[0], q[1], q[2], q[3])),
        })
    }
}
#[derive(Clone, Deserialize)]
struct Sample {
    timestamp_s: f64,
    image: PathBuf,
    #[serde(default)]
    truth: Option<PoseData>,
}
#[derive(Deserialize)]
struct Manifest {
    camera: Calibration,
    landmarks_m: Vec<[f64; 3]>,
    initial_pose: PoseData,
    temporal: bool,
    frames: Vec<Sample>,
}
#[derive(Serialize)]
struct ResultRow {
    timestamp_s: f64,
    image: PathBuf,
    state: &'static str,
    valid: bool,
    reason: &'static str,
    features: usize,
    matches: usize,
    position_m: Option<[f64; 3]>,
    quaternion_wxyz: Option<[f64; 4]>,
    position_error_m: Option<f64>,
    attitude_error_deg: Option<f64>,
    vision_us: u128,
}
#[derive(Serialize)]
struct Report {
    frames: usize,
    estimated: usize,
    valid: usize,
    scored: usize,
    position_rms_m: Option<f64>,
    attitude_rms_deg: Option<f64>,
    worst_position_m: Option<f64>,
    worst_attitude_deg: Option<f64>,
    vision_p95_us: u128,
    results: Vec<ResultRow>,
}

fn image(path: &Path, camera: Camera) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut decoder = png::Decoder::new(fs::File::open(path)?);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    if info.width as usize != camera.width || info.height as usize != camera.height {
        return Err(format!(
            "image dimensions do not match calibration: {}",
            path.display()
        )
        .into());
    }
    if reader.output_buffer_size() > 64 * 1024 * 1024 {
        return Err("image exceeds 64 MiB decode budget".into());
    }
    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer)?;
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => return Err("unsupported PNG color format".into()),
    };
    Ok(buffer[..info.buffer_size()]
        .chunks_exact(channels)
        .map(|p| {
            if channels < 3 {
                p[0]
            } else {
                (0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64) as u8
            }
        })
        .collect())
}
fn evaluate(manifest: &Manifest, root: &Path) -> Result<Report, Box<dyn Error>> {
    let c = &manifest.camera;
    if !c.distortion.iter().all(|v| v.is_finite() && *v == 0.0) {
        return Err("images must be rectified; nonzero distortion is unsupported".into());
    }
    let camera = Camera {
        width: c.width,
        height: c.height,
        fx: c.fx,
        fy: c.fy,
        cx: c.cx,
        cy: c.cy,
    };
    if camera.project(V3::new(0.0, 0.0, 1.0)).is_none()
        || manifest.frames.is_empty()
        || !(6..=32).contains(&manifest.landmarks_m.len())
    {
        return Err("invalid calibration, empty frames, or landmark count outside 6..32".into());
    }
    let model: Vec<_> = manifest
        .landmarks_m
        .iter()
        .map(|p| V3::from_row_slice(p))
        .collect();
    if model.iter().any(|p| !p.iter().all(|v| v.is_finite())) {
        return Err("nonfinite model landmarks".into());
    }
    let initial = manifest.initial_pose.pose()?;
    let mut nav = Navigator::default();
    let mut previous = None;
    let mut results = Vec::new();
    let (mut squared, mut angle_squared, mut scored, mut estimated, mut valid) =
        (0.0, 0.0, 0, 0, 0);
    let (mut worst, mut worst_angle) = (0.0_f64, 0.0_f64);
    let mut times = Vec::new();
    for sample in &manifest.frames {
        if !sample.timestamp_s.is_finite()
            || sample.timestamp_s < 0.0
            || previous.is_some_and(|p| sample.timestamp_s <= p)
        {
            return Err("timestamps must be finite, nonnegative, and strictly increasing; use independent mode for shuffled pose datasets".into());
        }
        previous = Some(sample.timestamp_s);
        if !manifest.temporal {
            nav = Navigator::default();
        }
        let pixels = image(&root.join(&sample.image), camera)?;
        nav.predict_to(sample.timestamp_s)
            .map_err(|r| format!("navigation clock: {}", r.label()))?;
        let start = Instant::now();
        let detected = corners::detect(&pixels, camera);
        let prior = nav.filter.as_ref().map(|f| &f.pose).unwrap_or(&initial);
        let matched = corners::associate(camera, &model, &detected, prior, 20.0);
        let solved = solve_robust(camera, &model, &matched, prior);
        let accepted = nav.observe(
            sample.timestamp_s,
            solved.as_ref().map(|p| &p.0),
            if matched.len() < 6 {
                Rejection::Association
            } else {
                Rejection::PoseFit
            },
        );
        let vision_us = start.elapsed().as_micros();
        times.push(vision_us);
        let health = nav.health();
        if health.valid {
            valid += 1;
        }
        let prediction = if !manifest.temporal && !accepted {
            None
        } else {
            nav.filter.as_ref().map(|f| f.pose.clone())
        };
        let mut row = ResultRow {
            timestamp_s: sample.timestamp_s,
            image: sample.image.clone(),
            state: health.state.label(),
            valid: health.valid,
            reason: health.rejection.label(),
            features: detected.len(),
            matches: matched.len(),
            position_m: None,
            quaternion_wxyz: None,
            position_error_m: None,
            attitude_error_deg: None,
            vision_us,
        };
        if let Some(pose) = prediction {
            estimated += 1;
            row.position_m = Some([pose.position.x, pose.position.y, pose.position.z]);
            let q = pose.attitude.quaternion();
            row.quaternion_wxyz = Some([q.w, q.i, q.j, q.k]);
            // Ground truth is deliberately first accessed here, after the estimator output is fixed.
            if let Some(truth) = &sample.truth {
                let truth = truth.pose()?;
                let e = (pose.position - truth.position).norm();
                let a = (pose.attitude * truth.attitude.inverse())
                    .angle()
                    .to_degrees();
                row.position_error_m = Some(e);
                row.attitude_error_deg = Some(a);
                squared += e * e;
                angle_squared += a * a;
                scored += 1;
                worst = worst.max(e);
                worst_angle = worst_angle.max(a);
            }
        }
        results.push(row);
    }
    times.sort_unstable();
    Ok(Report {
        frames: results.len(),
        estimated,
        valid,
        scored,
        position_rms_m: (scored > 0).then(|| (squared / scored as f64).sqrt()),
        attitude_rms_deg: (scored > 0).then(|| (angle_squared / scored as f64).sqrt()),
        worst_position_m: (scored > 0).then_some(worst),
        worst_attitude_deg: (scored > 0).then_some(worst_angle),
        vision_p95_us: times[((times.len() - 1) as f64 * 0.95).round() as usize],
        results,
    })
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: apsis-bench MANIFEST.json REPORT.json".into());
    }
    let path = Path::new(&args[1]);
    let manifest: Manifest = serde_json::from_str(&fs::read_to_string(path)?)?;
    let report = evaluate(&manifest, path.parent().unwrap_or(Path::new(".")))?;
    fs::write(&args[2], serde_json::to_string_pretty(&report)?)?;
    println!(
        "{} frames, {} estimates, {} valid, {} scored; failures remain in the report",
        report.frames, report.estimated, report.valid, report.scored
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn truth_labels_cannot_change_predictions() {
        let root = std::env::temp_dir().join(format!("apsis-bench-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let camera = Camera::new(320, 240, 500.0);
        let pose = apsis_sim::truth(0.0);
        let pixels = apsis_sim::scene::render(
            camera,
            &pose,
            0.0,
            apsis_sim::scene::Conditions::default(),
            &mut apsis_sim::Noise::new(42),
        );
        let file = fs::File::create(root.join("camera.png")).unwrap();
        let mut encoder = png::Encoder::new(file, 320, 240);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&pixels.gray)
            .unwrap();
        let q = Q::from_scaled_axis(V3::new(0.2, 0.3, 0.1));
        let q = q.quaternion();
        let mut m = Manifest {
            camera: Calibration {
                width: 320,
                height: 240,
                fx: 500.0,
                fy: 500.0,
                cx: 160.0,
                cy: 120.0,
                distortion: Vec::new(),
            },
            landmarks_m: apsis_core::math::landmarks()
                .iter()
                .map(|p| [p.x, p.y, p.z])
                .collect(),
            initial_pose: PoseData {
                position_m: [0.0, 0.0, 1.6],
                quaternion_wxyz: [q.w, q.i, q.j, q.k],
            },
            temporal: false,
            frames: vec![Sample {
                timestamp_s: 0.0,
                image: PathBuf::from("camera.png"),
                truth: Some(PoseData {
                    position_m: [0.0, 0.0, 1.6],
                    quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
                }),
            }],
        };
        let a = evaluate(&m, &root).unwrap();
        m.frames[0].truth.as_mut().unwrap().position_m = [50.0, 20.0, 30.0];
        let b = evaluate(&m, &root).unwrap();
        assert!(
            a.estimated > 0,
            "fixture must yield a pose, not a vacuous truth-isolation test"
        );
        assert_eq!(a.results[0].position_m, b.results[0].position_m);
        assert_eq!(a.results[0].quaternion_wxyz, b.results[0].quaternion_wxyz);
        assert_ne!(a.position_rms_m, b.position_rms_m);
    }
}
