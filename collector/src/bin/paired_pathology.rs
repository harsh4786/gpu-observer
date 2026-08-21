//! Cold, bounded paired-run analysis for EXP-0023.

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

const MAX_INPUT_BYTES: u64 = 1024 * 1024;
const MAX_PAIRS: usize = 30;

#[derive(Clone, Copy, Default)]
struct Arm {
    ttft_p99: f64,
    itl_p99: f64,
    throughput: f64,
    ttft_avg: f64,
    itl_avg: f64,
    duration: f64,
    mixed_steps: f64,
}

#[derive(Default)]
struct Pair {
    a: Option<Arm>,
    b: Option<Arm>,
}

struct Summary {
    a_mean: f64,
    b_mean: f64,
    delta_mean: f64,
    pct_mean: f64,
    pct_low: f64,
    pct_high: f64,
    t_statistic: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let program = PathBuf::from(args.next().unwrap_or_default());
    let usage = || {
        format!(
            "usage: {} RUN-METRICS.tsv PAIRED-SUMMARY.tsv",
            program.display()
        )
    };
    let input_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let output_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    if args.next().is_some() {
        return Err(usage().into());
    }
    let metadata = fs::metadata(&input_path)?;
    if metadata.len() == 0 || metadata.len() > MAX_INPUT_BYTES {
        return Err("run-metrics input violates its size bound".into());
    }
    let input = fs::read_to_string(&input_path)?;
    let mut lines = input.lines();
    let header = lines.next().ok_or("metrics file is empty")?;
    let expected = "pair\tarm\tordinal\tttft_p99_ms\titl_p99_ms\tthroughput_tok_s\tttft_avg_ms\titl_avg_ms\tduration_s\tmixed_steps";
    if header != expected {
        return Err("unexpected metrics schema".into());
    }
    let mut pairs = BTreeMap::<u32, Pair>::new();
    for (line_number, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 10 {
            return Err(format!("invalid field count on data line {}", line_number + 2).into());
        }
        let pair_id: u32 = fields[0].parse()?;
        let _: u32 = fields[2].parse()?;
        let arm = Arm {
            ttft_p99: fields[3].parse()?,
            itl_p99: fields[4].parse()?,
            throughput: fields[5].parse()?,
            ttft_avg: fields[6].parse()?,
            itl_avg: fields[7].parse()?,
            duration: fields[8].parse()?,
            mixed_steps: fields[9].parse()?,
        };
        let pair = pairs.entry(pair_id).or_default();
        let slot = match fields[1] {
            "A" => &mut pair.a,
            "B" => &mut pair.b,
            _ => return Err("arm must be A or B".into()),
        };
        if slot.replace(arm).is_some() {
            return Err("duplicate pair arm".into());
        }
    }
    if pairs.len() < 2 || pairs.len() > MAX_PAIRS {
        return Err("paired analysis requires 2..=30 complete pairs".into());
    }
    let mut complete = Vec::new();
    complete.try_reserve_exact(pairs.len())?;
    for (pair_id, pair) in pairs {
        complete.push((
            pair_id,
            pair.a.ok_or("pair is missing arm A")?,
            pair.b.ok_or("pair is missing arm B")?,
        ));
    }

    let metrics: [(&str, fn(Arm) -> f64); 7] = [
        ("ttft_p99_ms", |arm| arm.ttft_p99),
        ("itl_p99_ms", |arm| arm.itl_p99),
        ("throughput_tok_s", |arm| arm.throughput),
        ("ttft_avg_ms", |arm| arm.ttft_avg),
        ("itl_avg_ms", |arm| arm.itl_avg),
        ("duration_s", |arm| arm.duration),
        ("mixed_steps", |arm| arm.mixed_steps),
    ];
    let output = File::create(output_path)?;
    let mut writer = BufWriter::new(output);
    writeln!(
        writer,
        "metric\tpairs\tarm_a_mean\tarm_b_mean\tdelta_b_minus_a\tpaired_pct_mean\tci95_low_pct\tci95_high_pct\tt_statistic"
    )?;
    let mut ttft_low = f64::NAN;
    let mut mixed_low = f64::NAN;
    for (name, access) in metrics {
        let summary = summarize(&complete, access)?;
        writeln!(
            writer,
            "{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
            name,
            complete.len(),
            summary.a_mean,
            summary.b_mean,
            summary.delta_mean,
            summary.pct_mean,
            summary.pct_low,
            summary.pct_high,
            summary.t_statistic,
        )?;
        if name == "ttft_p99_ms" {
            ttft_low = summary.pct_low;
        } else if name == "mixed_steps" {
            mixed_low = summary.pct_low;
        }
    }
    writer.flush()?;
    let verdict = if ttft_low > 0.0 && mixed_low > 0.0 {
        "PASS"
    } else {
        "NOT_CONFIRMED"
    };
    println!(
        "summary status={} pairs={} criterion=ci95_lower_bound_positive_for_ttft_p99_and_mixed_steps ttft_p99_ci_low_pct={:.6} mixed_steps_ci_low_pct={:.6}",
        verdict,
        complete.len(),
        ttft_low,
        mixed_low,
    );
    Ok(())
}

fn summarize(pairs: &[(u32, Arm, Arm)], access: fn(Arm) -> f64) -> Result<Summary, Box<dyn Error>> {
    let n = pairs.len();
    let mut a_sum = 0.0;
    let mut b_sum = 0.0;
    let mut delta_sum = 0.0;
    let mut pct_sum = 0.0;
    for (_, a, b) in pairs {
        let a_value = access(*a);
        let b_value = access(*b);
        if !a_value.is_finite() || !b_value.is_finite() || a_value == 0.0 {
            return Err("metric contains a non-finite or zero baseline".into());
        }
        a_sum += a_value;
        b_sum += b_value;
        delta_sum += b_value - a_value;
        pct_sum += 100.0 * (b_value - a_value) / a_value;
    }
    let denominator = n as f64;
    let pct_mean = pct_sum / denominator;
    let mut squared = 0.0;
    for (_, a, b) in pairs {
        let value = 100.0 * (access(*b) - access(*a)) / access(*a);
        squared += (value - pct_mean) * (value - pct_mean);
    }
    let sample_sd = (squared / (denominator - 1.0)).sqrt();
    let standard_error = sample_sd / denominator.sqrt();
    let half_width = t_critical_975(n - 1)? * standard_error;
    let t_statistic = if standard_error == 0.0 {
        if pct_mean == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        pct_mean / standard_error
    };
    Ok(Summary {
        a_mean: a_sum / denominator,
        b_mean: b_sum / denominator,
        delta_mean: delta_sum / denominator,
        pct_mean,
        pct_low: pct_mean - half_width,
        pct_high: pct_mean + half_width,
        t_statistic,
    })
}

fn t_critical_975(df: usize) -> Result<f64, Box<dyn Error>> {
    const VALUES: [f64; 30] = [
        12.706_205, 4.302_653, 3.182_446, 2.776_445, 2.570_582, 2.446_912, 2.364_624, 2.306_004,
        2.262_157, 2.228_139, 2.200_985, 2.178_813, 2.160_369, 2.144_787, 2.131_450, 2.119_905,
        2.109_816, 2.100_922, 2.093_024, 2.085_963, 2.079_614, 2.073_873, 2.068_658, 2.063_899,
        2.059_539, 2.055_529, 2.051_831, 2.048_407, 2.045_230, 2.042_272,
    ];
    VALUES
        .get(df.wrapping_sub(1))
        .copied()
        .ok_or_else(|| "unsupported degrees of freedom".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_interval_uses_within_pair_percent_changes() {
        let pairs = [
            (
                1,
                Arm {
                    ttft_p99: 100.0,
                    ..Arm::default()
                },
                Arm {
                    ttft_p99: 110.0,
                    ..Arm::default()
                },
            ),
            (
                2,
                Arm {
                    ttft_p99: 200.0,
                    ..Arm::default()
                },
                Arm {
                    ttft_p99: 220.0,
                    ..Arm::default()
                },
            ),
        ];
        let summary = summarize(&pairs, |arm| arm.ttft_p99).unwrap();
        assert_eq!(summary.a_mean, 150.0);
        assert_eq!(summary.b_mean, 165.0);
        assert_eq!(summary.delta_mean, 15.0);
        assert!((summary.pct_mean - 10.0).abs() < 1e-12);
        assert!((summary.pct_low - 10.0).abs() < 1e-12);
        assert!((summary.pct_high - 10.0).abs() < 1e-12);
    }
}
