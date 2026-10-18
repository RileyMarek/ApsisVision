use apsis_core::math::{target, Camera, Pose, Q, V3};
use apsis_core::{vision, Filter};
use std::{error::Error, fmt::Write as _, fs, path::PathBuf, time::Instant};

struct Summary {
    rms: f64,
    attitude_rms: f64,
    accepted: usize,
    frames: usize,
}

fn run(frames: usize, seed: u64, out: PathBuf) -> Result<Summary, Box<dyn Error>> {
    fs::create_dir_all(&out)?;
    let camera = Camera {
        width: 640,
        height: 480,
        focal: 500.0,
    };
    let mut noise = apsis_sim::Noise::new(seed);
    let mut filter: Option<Filter> = None;
    let initial = Pose {
        position: V3::new(0.0, 0.0, 12.0),
        attitude: Q::identity(),
    };
    let mut csv=String::from("time_s,truth_x_m,truth_y_m,truth_z_m,estimate_x_m,estimate_y_m,estimate_z_m,position_error_m,attitude_error_deg,sigma_x_m,sigma_y_m,sigma_z_m,markers,accepted,latency_us,vx_m_s,vy_m_s,vz_m_s,qw,qx,qy,qz,wx_rad_s,wy_rad_s,wz_rad_s\n");
    let mut playback = String::from("[");
    let (mut squared, mut angle_squared, mut accepted) = (0.0, 0.0, 0);
    for frame in 0..frames {
        let start = Instant::now();
        let time = frame as f64 * 0.02;
        let truth = apsis_sim::truth(time);
        let dropout = (200..225).contains(&(frame % 500));
        let image = apsis_sim::render(camera, &truth, &mut noise, dropout);
        let points = vision::detect(&image, camera);
        if let Some(f) = &mut filter {
            f.predict(0.02);
        }
        let prior = filter.as_ref().map(|f| &f.pose).unwrap_or(&initial);
        let measurement = vision::solve(camera, &points, prior);
        let mut used = false;
        if let Some((pose, _)) = measurement {
            if let Some(f) = &mut filter {
                used = f.update(&pose);
            } else {
                filter = Some(Filter::new(pose));
                used = true;
            }
        }
        let f = filter.as_ref().ok_or("initial acquisition failed")?;
        accepted += usize::from(used);
        let error = (f.pose.position - truth.position).norm();
        let angle = (f.pose.attitude * truth.attitude.inverse())
            .angle()
            .to_degrees();
        squared += error * error;
        angle_squared += angle * angle;
        let p = f.pose.position;
        let sigmas = V3::new(
            f.covariance[(0, 0)].sqrt(),
            f.covariance[(1, 1)].sqrt(),
            f.covariance[(2, 2)].sqrt(),
        );
        let latency = start.elapsed().as_micros();
        let q = f.pose.attitude.quaternion();
        writeln!(csv,"{time:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{error:.6},{angle:.6},{:.6},{:.6},{:.6},{},{used},{latency},{:.6},{:.6},{:.6},{:.9},{:.9},{:.9},{:.9},{:.6},{:.6},{:.6}",truth.position.x,truth.position.y,truth.position.z,p.x,p.y,p.z,sigmas.x,sigmas.y,sigmas.z,points.len(),f.velocity.x,f.velocity.y,f.velocity.z,q.w,q.i,q.j,q.k,f.rate.x,f.rate.y,f.rate.z)?;
        if frame > 0 {
            playback.push(',');
        }
        write!(playback,"{{\"t\":{time:.4},\"error\":{error:.6},\"angle\":{angle:.6},\"used\":{used},\"sigma\":[{:.6},{:.6}],\"truth\":[",sigmas.x,sigmas.y)?;
        for (i, point) in target().iter().enumerate() {
            if i > 0 {
                playback.push(',');
            }
            let p = camera
                .project(truth.transform(*point))
                .ok_or("truth left camera")?;
            write!(playback, "[{:.3},{:.3}]", p.x, p.y)?;
        }
        playback.push_str("],\"estimate\":[");
        for (i, point) in target().iter().enumerate() {
            if i > 0 {
                playback.push(',');
            }
            let p = camera
                .project(f.pose.transform(*point))
                .ok_or("estimate left camera")?;
            write!(playback, "[{:.3},{:.3}]", p.x, p.y)?;
        }
        playback.push_str("],\"detected\":[");
        for (i, (_, point)) in points.iter().enumerate() {
            if i > 0 {
                playback.push(',');
            }
            write!(playback, "[{:.3},{:.3}]", point.x, point.y)?;
        }
        let center = camera.project(p).ok_or("estimated center left camera")?;
        write!(
            playback,
            "],\"center\":[{:.3},{:.3}],\"depth\":{:.6}}}",
            center.x, center.y, p.z
        )?;
    }
    playback.push(']');
    fs::write(out.join("telemetry.csv"), csv)?;
    fs::write(
        out.join("index.html"),
        include_str!("viewer.html").replace("__FRAMES__", &playback),
    )?;
    Ok(Summary {
        rms: (squared / frames as f64).sqrt(),
        attitude_rms: (angle_squared / frames as f64).sqrt(),
        accepted,
        frames,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut frames = 1500;
    let mut seed = 42;
    let mut out = PathBuf::from("output");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frames" => frames = args.next().ok_or("missing frame count")?.parse()?,
            "--seed" => seed = args.next().ok_or("missing seed")?.parse()?,
            "--out" => out = PathBuf::from(args.next().ok_or("missing output path")?),
            "--help" => {
                println!("apsis-cli [--frames 1500] [--seed 42] [--out output]");
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if !(1..=3000).contains(&frames) {
        return Err("frames must be between 1 and 3000 (validated 60-second scenario)".into());
    }
    let summary = run(frames, seed, out.clone())?;
    println!("{} frames at simulated 50 Hz; {} accepted poses\nPosition RMS: {:.4} m; attitude RMS: {:.3} deg\nOpen {}",summary.frames,summary.accepted,summary.rms,summary.attitude_rms,out.join("index.html").display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_to_filter_tracks_and_recovers_from_dropout() {
        for seed in [7, 42, 99] {
            let path =
                std::env::temp_dir().join(format!("apsis-test-{}-{seed}", std::process::id()));
            let s = run(600, seed, path).unwrap();
            assert!(s.rms < 0.15, "seed {seed}: position RMS {}", s.rms);
            assert!(
                s.attitude_rms < 3.0,
                "seed {seed}: attitude RMS {}",
                s.attitude_rms
            );
            assert!(s.accepted > 540, "accepted {}", s.accepted);
        }
    }
    #[test]
    fn full_scenario_exports_finite_states_and_recovers() {
        let path = std::env::temp_dir().join(format!("apsis-full-{}", std::process::id()));
        let s = run(3000, 42, path.clone()).unwrap();
        assert!(s.rms < 0.15, "position RMS {}", s.rms);
        assert!(s.attitude_rms < 3.0, "attitude RMS {}", s.attitude_rms);
        let csv = fs::read_to_string(path.join("telemetry.csv")).unwrap();
        let rows: Vec<_> = csv.lines().skip(1).collect();
        assert_eq!(rows.len(), 3000);
        for row in &rows {
            let values: Vec<_> = row.split(',').collect();
            assert_eq!(values.len(), 25);
            for (i, value) in values.iter().enumerate() {
                if i != 13 {
                    assert!(value.parse::<f64>().unwrap().is_finite());
                }
            }
        }
        assert_eq!(rows[210].split(',').nth(13), Some("false"));
        assert_eq!(rows[240].split(',').nth(13), Some("true"));
        assert!(!fs::read_to_string(path.join("index.html"))
            .unwrap()
            .contains("__FRAMES__"));
    }
}
