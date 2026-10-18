//! Bounded, streaming host adapter. Timing is host evidence, not a real-time guarantee.
use apsis_core::{
    math::{Pose, Q, V3},
    navigation::{Navigator, Rejection},
};
use serde::Deserialize;
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Read},
    time::{Duration, Instant},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    position_m: [f64; 3],
    quaternion_wxyz: [f64; 4],
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    captured_s: f64,
    delivered_s: f64,
    pose: Option<Observation>,
}

fn decode(line: &[u8]) -> Result<Packet, Box<dyn Error>> {
    let p: Packet = serde_json::from_slice(line)?;
    if !p.captured_s.is_finite()
        || !p.delivered_s.is_finite()
        || p.captured_s < 0.0
        || p.delivered_s < p.captured_s
    {
        return Err("invalid capture/delivery timestamp".into());
    }
    if let Some(o) = &p.pose {
        let norm = o.quaternion_wxyz.iter().map(|x| x * x).sum::<f64>();
        if !o
            .position_m
            .iter()
            .chain(o.quaternion_wxyz.iter())
            .all(|x| x.is_finite())
            || (norm - 1.0).abs() > 1e-6
            || o.position_m[2] <= 0.0
        {
            return Err("invalid camera pose or non-unit quaternion".into());
        }
    }
    Ok(p)
}

fn replay(reader: impl BufRead, paced: bool) -> Result<serde_json::Value, Box<dyn Error>> {
    let mut reader = reader;
    let mut nav = Navigator::default();
    let mut line = Vec::with_capacity(4097);
    let started = Instant::now();
    let mut origin = None;
    let mut last_delivery = None;
    let mut now = None;
    let mut packets = 0;
    let mut accepted = 0;
    let mut valid = 0;
    let mut late = 0;
    let mut deadline_misses = 0;
    let mut maximum_update_us = 0_u128;
    let mut maximum_scheduler_lateness_us = 0_u128;
    let mut reasons = std::collections::BTreeMap::<&str, usize>::new();
    loop {
        line.clear();
        let count = reader.by_ref().take(4097).read_until(b'\n', &mut line)?;
        if count == 0 {
            break;
        }
        if count > 4096 {
            return Err("record exceeds 4096-byte bound; stream stopped".into());
        }
        packets += 1;
        if packets > 100_000 {
            return Err("packet count exceeds 100000 bound".into());
        }
        let p = decode(&line)?;
        if last_delivery.is_some_and(|t| p.delivered_s <= t) {
            return Err("delivery clock repeated or reversed; stream stopped".into());
        }
        if last_delivery.is_some_and(|t| p.delivered_s - t > 10.0) {
            return Err("delivery gap exceeds 10-second adapter bound".into());
        }
        last_delivery = Some(p.delivered_s);
        let first = *origin.get_or_insert(p.delivered_s);
        let mut delivered = p.delivered_s;
        if paced {
            let deadline = Duration::from_secs_f64(p.delivered_s - first);
            if let Some(remaining) = deadline.checked_sub(started.elapsed()) {
                std::thread::sleep(remaining);
            }
            let lateness = started.elapsed().saturating_sub(deadline);
            maximum_scheduler_lateness_us = maximum_scheduler_lateness_us.max(lateness.as_micros());
            delivered += lateness.as_secs_f64();
        }
        let begin = Instant::now();
        if now.is_some_and(|t| delivered - t > 10.0 || delivered <= t) {
            return Err("scheduler exceeded the adapter clock bound; stream stopped".into());
        }
        // Bounded substeps preserve the 50 Hz propagation contract across missing frames.
        if let Some(previous) = now {
            let mut t = previous;
            while t + 0.02 < delivered - 1e-9 {
                t += 0.02;
                nav.predict_to(t).map_err(|r| r.label())?;
            }
        }
        nav.predict_to(delivered).map_err(|r| r.label())?;
        now = Some(delivered);
        let pose = p.pose.map(|o| {
            let [w, x, y, z] = o.quaternion_wxyz;
            Pose {
                position: V3::from_row_slice(&o.position_m),
                attitude: Q::new_normalize(apsis_core::math::Quaternion::new(w, x, y, z)),
            }
        });
        accepted += usize::from(nav.observe(p.captured_s, pose.as_ref(), Rejection::NoImage));
        let h = nav.health();
        valid += usize::from(h.valid);
        late += usize::from(h.rejection == Rejection::Stale);
        *reasons.entry(h.rejection.label()).or_default() += 1;
        let elapsed = begin.elapsed().as_micros();
        maximum_update_us = maximum_update_us.max(elapsed);
        deadline_misses += usize::from(elapsed > 20_000);
    }
    Ok(
        serde_json::json!({"platform":"host", "hardware_verified":false,"paced":paced,"packets":packets,"accepted":accepted,"valid_packet_samples":valid,"stale_packets":late,"rejections":reasons,"update_deadline_us":20000,"update_deadline_misses":deadline_misses,"maximum_update_us":maximum_update_us,"maximum_scheduler_lateness_us":maximum_scheduler_lateness_us,"bounded_input_bytes":4096,"maximum_packets":100000,"timing_scope":"adapter propagation and filter only; excludes image detection, parsing, file IO and waiting", "final_state":nav.health().state.label(),"final_valid":nav.health().valid}),
    )
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: apsis-replay PACKETS.jsonl [--paced]")?;
    let paced = match args.next().as_deref() {
        None => false,
        Some("--paced") => true,
        _ => return Err("unknown replay option".into()),
    };
    if args.next().is_some() {
        return Err("unexpected replay arguments".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&replay(
            BufReader::with_capacity(4096, File::open(path)?),
            paced
        )?)?
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn packet(c: f64, d: f64, pose: bool) -> String {
        format!(
            "{{\"captured_s\":{c},\"delivered_s\":{d},\"pose\":{}}}\n",
            if pose {
                "{\"position_m\":[0,0,2],\"quaternion_wxyz\":[1,0,0,0]}"
            } else {
                "null"
            }
        )
    }
    #[test]
    fn late_duplicate_capture_outage_and_recovery_are_observable() {
        let data = packet(0.0, 0.0, true)
            + &packet(0.1, 0.1, true)
            + &packet(0.2, 0.2, true)
            + &packet(0.2, 0.3, true)
            + &packet(0.4, 0.5, true)
            + &packet(3.5, 3.5, false)
            + &packet(3.6, 3.6, true)
            + &packet(3.7, 3.7, true)
            + &packet(3.8, 3.8, true)
            + &packet(4.2, 4.2, true);
        let r = replay(data.as_bytes(), false).unwrap();
        assert_eq!(r["stale_packets"], 1);
        assert_eq!(r["rejections"]["out_of_order"], 1);
        assert_eq!(r["rejections"]["camera_unavailable"], 1);
        assert_eq!(r["final_valid"], true);
    }
    #[test]
    fn corrupt_nonunit_oversize_and_reversed_delivery_stop_the_stream() {
        assert!(decode(b"{}").is_err());
        assert!(decode(packet(1.0, 0.0, true).as_bytes()).is_err());
        assert!(decode(
            packet(0.0, 0.0, true)
                .replace("[1,0,0,0]", "[2,0,0,0]")
                .as_bytes()
        )
        .is_err());
        assert!(replay(vec![b'x'; 4097].as_slice(), false).is_err());
        let data = packet(0.0, 0.0, true) + &packet(0.0, 0.0, true);
        assert!(replay(data.as_bytes(), false).is_err());
    }
}
