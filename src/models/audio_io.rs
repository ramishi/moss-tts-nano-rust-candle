use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};

/// Detect audio format by reading the first few bytes (magic bytes).
fn detect_audio_format(path: &Path) -> Result<AudioFormat> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open audio file: {}", path.display()))?;
    let mut magic = [0u8; 4];
    use std::io::Read;
    file.read_exact(&mut magic)
        .with_context(|| format!("Cannot read audio file header: {}", path.display()))?;

    if &magic[0..4] == b"RIFF" {
        Ok(AudioFormat::Wav)
    } else if &magic[0..4] == b"fLaC" {
        Ok(AudioFormat::Flac)
    } else if &magic[0..3] == b"ID3" || (magic[0] == 0xFF && (magic[1] & 0xE0) == 0xE0) {
        Ok(AudioFormat::Mp3)
    } else {
        Ok(AudioFormat::Unknown)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AudioFormat {
    Wav,
    Flac,
    Mp3,
    Unknown,
}

/// Load an audio file (WAV, FLAC, or MP3), resample to target rate, convert to target channels.
/// Returns waveform as Tensor of shape (1, target_channels, T) on the given device.
///
/// Matches Python `_load_reference_audio()`:
///   1. Read audio and decode to f32 samples
///   2. Convert to f32 planar (C, T)
///   3. Resample if needed (using rubato SincFixedIn)
///   4. Channel conversion (mono→stereo: repeat; stereo→mono: average)
///   5. Return Tensor (1, C, T)
pub fn load_and_prepare_wav(
    path: &Path,
    target_sample_rate: usize,
    target_channels: usize,
    device: &Device,
) -> Result<Tensor> {
    let path_str = path.to_str().context("Invalid path")?;

    let format = detect_audio_format(path)?;

    // Decode to interleaved f32 samples + (sample_rate, channels)
    let (samples, current_sample_rate, current_channels) = match format {
        AudioFormat::Wav => decode_wav_hound(path_str)?,
        AudioFormat::Flac | AudioFormat::Mp3 | AudioFormat::Unknown => decode_symphonia(path)?,
    };

    if samples.is_empty() {
        anyhow::bail!("Audio file is empty: {}", path_str);
    }

    // Convert interleaved to planar (C, T)
    let total_frames = samples.len() / current_channels;
    let mut planar = vec![vec![0.0f32; total_frames]; current_channels];
    for (i, &sample) in samples.iter().enumerate() {
        let ch = i % current_channels;
        let frame = i / current_channels;
        planar[ch][frame] = sample;
    }

    // Resample if needed
    let planar_data = if current_sample_rate != target_sample_rate {
        resample_channels(&planar, current_sample_rate, target_sample_rate)?
    } else {
        planar
    };

    // Channel conversion
    let converted = convert_channels(&planar_data, planar_data.len(), target_channels)?;
    let final_channels = converted.len();
    let final_frames = converted[0].len();

    // Build Tensor (1, C, T)
    let mut flat: Vec<f32> = Vec::with_capacity(final_channels * final_frames);
    for ch_data in &converted {
        flat.extend_from_slice(ch_data);
    }

    let tensor = Tensor::from_slice(&flat, (1, final_channels, final_frames), device)?;

    Ok(tensor)
}

/// Decode a RIFF WAV file using hound (original path).
fn decode_wav_hound(path_str: &str) -> Result<(Vec<f32>, usize, usize)> {
    let reader = hound::WavReader::open(path_str)
        .with_context(|| format!("Failed to open WAV file: {}", path_str))?;

    let spec = reader.spec();
    let current_sample_rate = spec.sample_rate as usize;
    let current_channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to read float samples")?,
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to read int samples")?
        }
    };

    Ok((samples, current_sample_rate, current_channels))
}

/// Decode any supported audio file (FLAC, MP3, WAV, etc.) using symphonia.
fn decode_symphonia(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
    use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open audio file: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .with_context(|| format!("Failed to probe audio format: {}", path.display()))?;

    let mut format_reader = probed.format;

    let track = format_reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("No supported audio track found in: {}", path.display()))?;

    let track_id = track.id;
    let codec_params = &track.codec_params;
    let current_sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("Cannot determine sample rate: {}", path.display()))?
        as usize;
    let current_channels = codec_params
        .channels
        .ok_or_else(|| anyhow::anyhow!("Cannot determine channel count: {}", path.display()))?
        .count();

    let decoder = symphonia::default::get_codecs()
        .make(codec_params, &decoder_opts)
        .with_context(|| format!("Failed to create decoder for: {}", path.display()))?;

    let mut decoder = decoder;
    let mut interleaved = Vec::new();

    loop {
        let packet = match format_reader.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Decode error: {}", e));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                append_interleaved_from_ref(&mut interleaved, &decoded, current_channels);
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow::anyhow!("Decode error: {}", e)),
        }
    }

    Ok((interleaved, current_sample_rate, current_channels))
}

/// Append samples from a symphonia AudioBufferRef to an interleaved f32 Vec.
fn append_interleaved_from_ref(out: &mut Vec<f32>, buf_ref: &AudioBufferRef, n_channels: usize) {
    let n_frames = buf_ref.frames();
    let start = out.len();
    out.resize(start + n_frames * n_channels, 0.0f32);

    // Convert each sample type to f32, handling interleaved layout
    macro_rules! copy_interleaved {
        ($buf:expr, $convert:expr) => {{
            for ch_idx in 0..n_channels {
                let channel = $buf.chan(ch_idx);
                for (frame_idx, sample) in channel.iter().enumerate() {
                    out[start + frame_idx * n_channels + ch_idx] = $convert(*sample);
                }
            }
        }};
    }

    match buf_ref {
        AudioBufferRef::U8(buf) => copy_interleaved!(buf, |s: u8| (s as f32 - 128.0) / 128.0),
        AudioBufferRef::U16(buf) => copy_interleaved!(buf, |s: u16| (s as f32 - 32768.0) / 32768.0),
        AudioBufferRef::U24(buf) => copy_interleaved!(buf, |s: symphonia::core::sample::u24| (s.0
            as f32
            - 8388608.0)
            / 8388608.0),
        AudioBufferRef::U32(buf) => {
            copy_interleaved!(buf, |s: u32| (s as f32 - 2147483648.0) / 2147483648.0)
        }
        AudioBufferRef::S8(buf) => copy_interleaved!(buf, |s: i8| s as f32 / 128.0),
        AudioBufferRef::S16(buf) => copy_interleaved!(buf, |s: i16| s as f32 / 32768.0),
        AudioBufferRef::S24(buf) => copy_interleaved!(buf, |s: symphonia::core::sample::i24| s.0
            as f32
            / 8388608.0),
        AudioBufferRef::S32(buf) => copy_interleaved!(buf, |s: i32| s as f32 / 2147483648.0),
        AudioBufferRef::F32(buf) => copy_interleaved!(buf, |s: f32| s),
        AudioBufferRef::F64(buf) => copy_interleaved!(buf, |s: f64| s as f32),
    }
}

/// Resample each channel using rubato SincFixedIn.
/// Handles the case where input length is not a multiple of chunk_size by padding.
fn resample_channels(
    channels: &[Vec<f32>],
    from_rate: usize,
    to_rate: usize,
) -> Result<Vec<Vec<f32>>> {
    let chunk_size = 1024;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 128,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::new(
        to_rate as f64 / from_rate as f64,
        1.0, // max_resample_ratio_relative (fixed ratio)
        params,
        chunk_size,
        channels.len(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create resampler: {}", e))?;

    let input_len = channels[0].len();
    let ratio = to_rate as f64 / from_rate as f64;
    let expected_output_len = (input_len as f64 * ratio).round() as usize;

    // Pad input channels to multiple of chunk_size
    let padded_len = input_len.div_ceil(chunk_size) * chunk_size;
    let mut padded_channels: Vec<Vec<f32>> = channels.to_vec();
    for ch in &mut padded_channels {
        ch.resize(padded_len, 0.0);
    }

    // Process in chunks
    let mut all_output: Vec<Vec<f32>> = vec![Vec::new(); channels.len()];
    for chunk_start in (0..padded_len).step_by(chunk_size) {
        let chunk: Vec<Vec<f32>> = padded_channels
            .iter()
            .map(|ch| ch[chunk_start..chunk_start + chunk_size].to_vec())
            .collect();

        let resampled = resampler
            .process(&chunk, None)
            .map_err(|e| anyhow::anyhow!("Resampling failed: {}", e))?;

        for (i, ch_out) in resampled.into_iter().enumerate() {
            all_output[i].extend_from_slice(&ch_out);
        }
    }

    // Trim to expected output length
    for ch in &mut all_output {
        ch.truncate(expected_output_len);
    }

    Ok(all_output)
}

/// Channel conversion matching Python logic.
fn convert_channels(
    channels: &[Vec<f32>],
    from_channels: usize,
    to_channels: usize,
) -> Result<Vec<Vec<f32>>> {
    if from_channels == to_channels {
        Ok(channels.to_vec())
    } else if from_channels == 1 && to_channels > 1 {
        let mono = &channels[0];
        Ok((0..to_channels).map(|_| mono.clone()).collect())
    } else if from_channels > 1 && to_channels == 1 {
        let frames = channels[0].len();
        let mut mono = vec![0.0f32; frames];
        for ch_data in channels {
            for (i, &s) in ch_data.iter().enumerate() {
                mono[i] += s;
            }
        }
        let n = channels.len() as f32;
        for s in &mut mono {
            *s /= n;
        }
        Ok(vec![mono])
    } else {
        anyhow::bail!(
            "Unsupported channel conversion: {} -> {}",
            from_channels,
            to_channels
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;
    use tempfile::TempDir;

    fn write_test_wav(
        samples: &[f32],
        sample_rate: u32,
        channels: u16,
        bits: u16,
    ) -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.wav");
        let file = std::fs::File::create(&path).unwrap();
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: bits,
            sample_format: if bits == 32 {
                hound::SampleFormat::Float
            } else {
                hound::SampleFormat::Int
            },
        };
        let mut writer = hound::WavWriter::new(file, spec).unwrap();
        if bits == 32 {
            for &s in samples {
                writer.write_sample(s).unwrap();
            }
        } else {
            let max_val = (1i32 << (bits - 1)) - 1;
            for &s in samples {
                writer.write_sample((s * max_val as f32) as i32).unwrap();
            }
        }
        writer.finalize().unwrap();
        (dir, path)
    }

    #[test]
    fn test_load_mono_wav_resample() {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) * 2.0 - 1.0).collect();
        let (_dir, path) = write_test_wav(&samples, 16000, 1, 16);

        let device = Device::Cpu;
        let tensor = load_and_prepare_wav(&path, 48000, 2, &device).unwrap();

        let (b, c, t) = tensor.dims3().unwrap();
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert!(t > 250 && t < 350, "Expected ~300 frames, got {}", t);
    }

    #[test]
    fn test_load_no_conversion_needed() {
        let samples: Vec<f32> = (0..480).map(|i| (i as f32 / 480.0)).collect();
        let (_dir, path) = write_test_wav(&samples, 48000, 2, 32);

        let device = Device::Cpu;
        let tensor = load_and_prepare_wav(&path, 48000, 2, &device).unwrap();

        let (b, c, t) = tensor.dims3().unwrap();
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert_eq!(t, 240);
    }

    #[test]
    fn test_stereo_to_mono() {
        let samples: Vec<f32> = (0..200).map(|i| (i as f32 / 200.0)).collect();
        let (_dir, path) = write_test_wav(&samples, 48000, 2, 32);

        let device = Device::Cpu;
        let tensor = load_and_prepare_wav(&path, 48000, 1, &device).unwrap();

        let (b, c, t) = tensor.dims3().unwrap();
        assert_eq!(b, 1);
        assert_eq!(c, 1);
        assert_eq!(t, 100);
    }

    #[test]
    fn test_channel_values_correct() {
        let mut samples = Vec::new();
        for _ in 0..100 {
            samples.push(0.5);
            samples.push(-0.5);
        }
        let (_dir, path) = write_test_wav(&samples, 48000, 2, 32);

        let device = Device::Cpu;
        let tensor = load_and_prepare_wav(&path, 48000, 2, &device).unwrap();

        let flat = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for i in 0..100 {
            assert!((flat[i] - 0.5).abs() < 1e-6, "ch0[{}] = {}", i, flat[i]);
            assert!(
                (flat[100 + i] - (-0.5)).abs() < 1e-6,
                "ch1[{}] = {}",
                i,
                flat[100 + i]
            );
        }
    }
}
