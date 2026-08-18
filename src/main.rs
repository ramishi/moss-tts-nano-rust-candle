use anyhow::{Context, Result};
use candle_core::Device;
use clap::Parser;
use hf_hub::api::sync::Api;
use moss_tts_nano_rust_candle::pipeline::{GenerateParams, MossTTS, SAMPLE_RATE};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const HF_REPO: &str = "ramishi/moss-tts-nano-candle";

// GitHub Release fallback: weights mirrored from HuggingFace.
// Change this tag when you publish a new release via scripts/push_release.sh.
const GH_REPO: &str = "ramishi/moss-tts-nano-rust-candle";
const GH_RELEASE_TAG: &str = "v0.1.0-models";

// Embedded manifest: SHA256 + size for every model file.
// This is the source of truth for download verification.
// Generated from release/weights_manifest.json — keep in sync.
#[derive(serde::Deserialize)]
struct ManifestFile {
    size: u64,
    sha256: String,
    // Used by scripts/push_release.sh to decide which files to upload,
    // not read by the binary itself.
    #[serde(default)]
    #[allow(dead_code)]
    github_release_asset: bool,
}

#[derive(serde::Deserialize)]
struct Manifest {
    files: std::collections::HashMap<String, ManifestFile>,
}

const MANIFEST_JSON: &str = include_str!("../release/weights_manifest.json");

/// Get the cache directory for GitHub Release downloads.
/// Uses platform conventions: ~/.cache/moss-tts-nano-rust-candle/ on Linux,
/// ~/Library/Caches/moss-tts-nano-rust-candle/ on macOS, %LOCALAPPDATA% on Windows.
fn gh_cache_dir() -> Result<PathBuf> {
    let dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("moss-tts-nano-rust-candle");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Compute SHA256 of a file.
fn file_sha256(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024]; // 1 MB buffer
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Download a file from a GitHub Release, verifying SHA256 against the manifest.
/// Returns the local path of the verified file.
fn download_from_github_release(filename: &str) -> Result<PathBuf> {
    let manifest: Manifest = serde_json::from_str(MANIFEST_JSON)
        .context("Failed to parse embedded weights_manifest.json")?;

    let file_info = manifest.files.get(filename).ok_or_else(|| {
        anyhow::anyhow!(
            "File '{}' not found in embedded weights_manifest.json",
            filename
        )
    })?;

    let cache_dir = gh_cache_dir()?;
    let local_path = cache_dir.join(filename);

    // Check if already cached and valid
    if local_path.exists() {
        let actual_size = std::fs::metadata(&local_path)?.len();
        if actual_size == file_info.size {
            println!("[GH] Verifying cached: {}", filename);
            let actual_sha = file_sha256(&local_path)?;
            if actual_sha == file_info.sha256 {
                println!("[GH] Cache valid: {} -> {}", filename, local_path.display());
                return Ok(local_path);
            } else {
                println!(
                    "[GH] Cache SHA256 mismatch (expected {}, got {}), re-downloading",
                    &file_info.sha256[..16],
                    &actual_sha[..16]
                );
            }
        } else {
            println!(
                "[GH] Cache size mismatch (expected {}, got {}), re-downloading",
                file_info.size, actual_size
            );
        }
        // Remove invalid cache
        let _ = std::fs::remove_file(&local_path);
    }

    // Download from GitHub Release
    let url = format!(
        "https://github.com/{}/releases/download/{}/{}",
        GH_REPO, GH_RELEASE_TAG, filename
    );
    println!("[GH] Downloading: {}", url);
    println!(
        "[GH] Expected size: {} bytes ({:.1} MB), SHA256: {}...",
        file_info.size,
        file_info.size as f64 / (1024.0 * 1024.0),
        &file_info.sha256[..16]
    );

    // Download to a temp file first, then verify + rename
    let tmp_path = local_path.with_extension("safetensors.tmp");
    if tmp_path.exists() {
        let _ = std::fs::remove_file(&tmp_path);
    }

    let response = ureq::get(&url)
        .call()
        .map_err(|e| anyhow::anyhow!("GitHub Release download failed for '{}': {}", filename, e))?;

    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(&tmp_path)?;
    std::io::copy(&mut reader, &mut file)?;
    file.sync_all()?;
    drop(file);

    // Verify size
    let actual_size = std::fs::metadata(&tmp_path)?.len();
    if actual_size != file_info.size {
        let _ = std::fs::remove_file(&tmp_path);
        anyhow::bail!(
            "Downloaded size mismatch for '{}': expected {}, got {}",
            filename,
            file_info.size,
            actual_size
        );
    }

    // Verify SHA256
    println!("[GH] Verifying SHA256...");
    let actual_sha = file_sha256(&tmp_path)?;
    if actual_sha != file_info.sha256 {
        let _ = std::fs::remove_file(&tmp_path);
        anyhow::bail!(
            "SHA256 mismatch for '{}':\n  expected: {}\n  actual:   {}",
            filename,
            file_info.sha256,
            actual_sha
        );
    }

    // Rename temp → final
    std::fs::rename(&tmp_path, &local_path)?;
    println!("[GH] Verified: {} -> {}", filename, local_path.display());
    Ok(local_path)
}

/// Resolve a model file: try HF → GitHub Release → local path → backup dir.
fn resolve_model_path(filename: &str, local_default: &Path) -> Result<PathBuf> {
    // First try: download from HF (uses cache automatically)
    if let Ok(api) = Api::new() {
        let repo = api.model(HF_REPO.to_string());
        match repo.get(filename) {
            Ok(path) => {
                println!("[HF] Downloaded/cached: {} -> {}", filename, path.display());
                return Ok(path);
            }
            Err(e) => {
                println!(
                    "[LOCAL] HF download failed for '{}': {}, trying GitHub Release fallback",
                    filename, e
                );
            }
        }
    } else {
        println!("[LOCAL] HF API init failed, trying GitHub Release fallback");
    }

    // Second try: GitHub Release (with SHA256 verification)
    match download_from_github_release(filename) {
        Ok(path) => return Ok(path),
        Err(e) => {
            println!(
                "[LOCAL] GitHub Release download failed for '{}': {}, trying local path",
                filename, e
            );
        }
    }

    // Third try: local path as-is
    if local_default.exists() {
        return Ok(local_default.to_path_buf());
    }

    // Fourth try: models_backup_YYYYMMDDHHMMSS/ directory
    let parent = local_default.parent().unwrap_or(std::path::Path::new("."));
    let backup_dir = parent.parent().unwrap_or(parent);

    let mut best_backup: Option<PathBuf> = None;
    let mut best_name: Option<String> = None;

    if let Ok(entries) = std::fs::read_dir(backup_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("models_backup_") && entry.path().is_dir() {
                if best_name.as_ref().is_none() || &name > best_name.as_ref().unwrap() {
                    best_backup = Some(entry.path());
                    best_name = Some(name);
                }
            }
        }
    }

    if let Some(bak) = best_backup {
        let subdir = parent
            .file_name()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let candidate = bak.join(&subdir).join(
            local_default
                .file_name()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
        );
        if candidate.exists() {
            println!("[BACKUP] Using model from backup: {}", candidate.display());
            return Ok(candidate);
        }
    }

    anyhow::bail!(
        "Model file '{}' not found.\n\
         Tried: HF cache ({}), GitHub Release ({}/{}), and local path.\n\
         Ensure internet connection or provide explicit local paths via --lm-weights etc.",
        filename,
        HF_REPO,
        GH_REPO,
        GH_RELEASE_TAG
    )
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    text: String,

    #[arg(short, long, default_value = "output.wav")]
    output: String,

    #[arg(long, default_value = "voice_clone")]
    mode: String,

    #[arg(long)]
    prompt_audio_path: Option<PathBuf>,

    #[arg(long)]
    tokenizer: Option<PathBuf>,

    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long)]
    lm_weights: Option<PathBuf>,

    #[arg(long)]
    codec_config: Option<PathBuf>,

    #[arg(long)]
    codec_weights: Option<PathBuf>,

    #[arg(long, default_value_t = 1.2)]
    audio_repetition_penalty: f64,

    #[arg(long, default_value_t = 375)]
    max_frames: usize,

    #[arg(long, default_value_t = false)]
    cpu: bool,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    do_sample: bool,

    #[arg(long, default_value_t = true)]
    normalize_text: bool,

    #[arg(long, default_value_t = 40)]
    seed: u64,

    #[arg(long, default_value_t = 1.0)]
    text_temperature: f64,

    #[arg(long, default_value_t = 1.0)]
    text_top_p: f64,

    #[arg(long, default_value_t = 50)]
    text_top_k: usize,

    #[arg(long, default_value_t = 1.0)]
    audio_temperature: f64,

    #[arg(long, default_value_t = 0.95)]
    audio_top_p: f64,

    #[arg(long, default_value_t = 25)]
    audio_top_k: usize,

    #[arg(long)]
    export_latents: Option<PathBuf>,

    #[arg(long, default_value_t = 35)]
    voice_clone_max_text_tokens: usize,

    #[arg(long, default_value_t = false)]
    print_chunks: bool,

    /// Skip all downloads (HF + GitHub Release), use only local file paths.
    /// Provide explicit --lm-weights, --config, etc. when using this flag.
    #[arg(long, default_value_t = false)]
    local_only: bool,

    /// Skip HuggingFace download, fall back to GitHub Release only.
    /// Use this when HF is blocked/unavailable but GitHub is reachable.
    #[arg(long, default_value_t = false)]
    github_only: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate mode
    let mode = args.mode.to_lowercase();
    if mode != "voice_clone" && mode != "continuation" {
        anyhow::bail!(
            "Unsupported mode '{}'. Use 'voice_clone' or 'continuation'.",
            args.mode
        );
    }
    if mode == "voice_clone" && args.prompt_audio_path.is_none() {
        anyhow::bail!("voice_clone mode requires --prompt-audio-path");
    }

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };

    let default_lm_dir = PathBuf::from("models/MOSS-TTS-Nano");
    let default_codec_dir = PathBuf::from("models/MOSS-Audio-Tokenizer-Nano");

    let tts = if args.local_only {
        println!("[LOCAL-ONLY] Skipping all downloads, using explicit local paths only.");

        let tokenizer = args
            .tokenizer
            .clone()
            .unwrap_or_else(|| default_lm_dir.join("tokenizer.model"));
        let config = args
            .config
            .clone()
            .unwrap_or_else(|| default_lm_dir.join("config.json"));
        let lm_weights = args
            .lm_weights
            .clone()
            .unwrap_or_else(|| default_lm_dir.join("moss_tts_nano_lm.safetensors"));
        let codec_config = args
            .codec_config
            .clone()
            .unwrap_or_else(|| default_codec_dir.join("moss_audio_tokenizer_config.json"));
        let codec_weights = args
            .codec_weights
            .clone()
            .unwrap_or_else(|| default_codec_dir.join("moss_audio_tokenizer.safetensors"));

        for (name, path) in [
            ("Tokenizer", &tokenizer),
            ("LM config", &config),
            ("LM weights", &lm_weights),
            ("Codec config", &codec_config),
            ("Codec weights", &codec_weights),
        ] {
            if !path.exists() {
                anyhow::bail!("{} not found: {}", name, path.display());
            }
        }

        MossTTS::load(
            &tokenizer,
            &config,
            &lm_weights,
            &codec_config,
            &codec_weights,
            &device,
        )
        .context("Failed to load models")?
    } else if args.github_only {
        // --github-only: skip HF, download directly from GitHub Release with SHA256 verification
        println!("[GITHUB-ONLY] Skipping HuggingFace, downloading from GitHub Release {}...", GH_RELEASE_TAG);

        let tokenizer = download_from_github_release("tokenizer.model")
            .or_else(|e| {
                let local = args.tokenizer.clone().unwrap_or_else(|| default_lm_dir.join("tokenizer.model"));
                if local.exists() {
                    println!("[LOCAL] GitHub download failed for tokenizer.model ({}), using local: {}", e, local.display());
                    Ok(local)
                } else {
                    Err(e)
                }
            })?;
        let config = download_from_github_release("config.json")
            .or_else(|e| {
                let local = args.config.clone().unwrap_or_else(|| default_lm_dir.join("config.json"));
                if local.exists() {
                    println!("[LOCAL] GitHub download failed for config.json ({}), using local: {}", e, local.display());
                    Ok(local)
                } else {
                    Err(e)
                }
            })?;
        let lm_weights = download_from_github_release("moss_tts_nano_lm.safetensors")
            .or_else(|e| {
                let local = args.lm_weights.clone().unwrap_or_else(|| default_lm_dir.join("moss_tts_nano_lm.safetensors"));
                if local.exists() {
                    println!("[LOCAL] GitHub download failed for lm weights ({}), using local: {}", e, local.display());
                    Ok(local)
                } else {
                    Err(e)
                }
            })?;
        let codec_config = download_from_github_release("moss_audio_tokenizer_config.json")
            .or_else(|e| {
                let local = args.codec_config.clone().unwrap_or_else(|| default_codec_dir.join("moss_audio_tokenizer_config.json"));
                if local.exists() {
                    println!("[LOCAL] GitHub download failed for codec config ({}), using local: {}", e, local.display());
                    Ok(local)
                } else {
                    Err(e)
                }
            })?;
        let codec_weights = download_from_github_release("moss_audio_tokenizer.safetensors")
            .or_else(|e| {
                let local = args.codec_weights.clone().unwrap_or_else(|| default_codec_dir.join("moss_audio_tokenizer.safetensors"));
                if local.exists() {
                    println!("[LOCAL] GitHub download failed for codec weights ({}), using local: {}", e, local.display());
                    Ok(local)
                } else {
                    Err(e)
                }
            })?;

        println!("Loading models on {:?}...", device);
        println!("  tokenizer  = {}", tokenizer.display());
        println!("  config     = {}", config.display());
        println!("  lm_weights = {}", lm_weights.display());
        println!("  codec_cfg  = {}", codec_config.display());
        println!("  codec_wt   = {}", codec_weights.display());

        MossTTS::load(
            &tokenizer,
            &config,
            &lm_weights,
            &codec_config,
            &codec_weights,
            &device,
        )
        .context("Failed to load models")?
    } else {
        // Resolve model paths: HF cache → GitHub Release → local → backup
        println!("Resolving model files...");

        let tokenizer = args
            .tokenizer
            .clone()
            .unwrap_or_else(|| default_lm_dir.join("tokenizer.model"));
        let config = args
            .config
            .clone()
            .unwrap_or_else(|| default_lm_dir.join("config.json"));
        let lm_weights = args
            .lm_weights
            .clone()
            .unwrap_or_else(|| default_lm_dir.join("moss_tts_nano_lm.safetensors"));
        let codec_config = args
            .codec_config
            .clone()
            .unwrap_or_else(|| default_codec_dir.join("moss_audio_tokenizer_config.json"));
        let codec_weights = args
            .codec_weights
            .clone()
            .unwrap_or_else(|| default_codec_dir.join("moss_audio_tokenizer.safetensors"));

        let tokenizer = resolve_model_path("tokenizer.model", &tokenizer)?;
        let config = resolve_model_path("config.json", &config)?;
        let lm_weights = resolve_model_path("moss_tts_nano_lm.safetensors", &lm_weights)?;
        let codec_config = resolve_model_path("moss_audio_tokenizer_config.json", &codec_config)?;
        let codec_weights = resolve_model_path("moss_audio_tokenizer.safetensors", &codec_weights)?;

        println!("Loading models on {:?}...", device);
        println!("  tokenizer  = {}", tokenizer.display());
        println!("  config     = {}", config.display());
        println!("  lm_weights = {}", lm_weights.display());
        println!("  codec_cfg  = {}", codec_config.display());
        println!("  codec_wt   = {}", codec_weights.display());

        MossTTS::load(
            &tokenizer,
            &config,
            &lm_weights,
            &codec_config,
            &codec_weights,
            &device,
        )
        .context("Failed to load models")?
    };

    run(args, tts)
}

fn run(args: Args, tts: MossTTS) -> Result<()> {
    let mode = args.mode.to_lowercase();

    // --print-chunks: preview chunks and exit
    if args.print_chunks && mode == "voice_clone" {
        let normalized_text = if args.normalize_text {
            moss_tts_nano_rust_candle::models::text_normalize::prepare_text_for_sentence_chunking(
                &args.text,
            )
        } else {
            args.text.clone()
        };

        let count_fn = |t: &str| -> usize {
            tts.tokenizer_ref()
                .encode(t)
                .map(|v: Vec<u32>| v.len())
                .unwrap_or(0)
        };
        let chunks = moss_tts_nano_rust_candle::models::text_normalize::split_into_best_sentences(
            &normalized_text,
            args.voice_clone_max_text_tokens,
            &count_fn,
        );
        println!("Voice clone text chunks");
        println!("----------------------");
        println!(
            "max_tokens={} chunks={}",
            args.voice_clone_max_text_tokens,
            chunks.len()
        );
        for (i, chunk) in chunks.iter().enumerate() {
            println!("[chunk {}]", i + 1);
            println!("{}", chunk);
            println!();
        }
        return Ok(());
    }

    let params = GenerateParams {
        max_frames: args.max_frames,
        do_sample: args.do_sample,
        text_temperature: args.text_temperature,
        text_top_p: args.text_top_p,
        text_top_k: args.text_top_k,
        audio_temperature: args.audio_temperature,
        audio_top_p: args.audio_top_p,
        audio_top_k: args.audio_top_k,
        audio_repetition_penalty: args.audio_repetition_penalty,
        normalize_text: args.normalize_text,
        seed: Some(args.seed),
        export_dir: args.export_latents.clone(),
        voice_clone_max_text_tokens: args.voice_clone_max_text_tokens,
    };

    println!("Generating speech for: \"{}\" [mode={}]", args.text, mode);
    let start = std::time::Instant::now();

    let waveform = match mode.as_str() {
        "voice_clone" => {
            let prompt_path = args.prompt_audio_path.as_ref().unwrap();
            tts.voice_clone(&args.text, prompt_path, &params)?
        }
        _ => tts.text_to_speech(&args.text, &params)?,
    };
    let duration = start.elapsed();

    // Save WAV
    let dims = waveform.dims();
    let num_channels = if dims.len() == 2 { dims[0] } else { 1 };
    let num_frames = if dims.len() == 2 { dims[1] } else { dims[0] };

    println!(
        "Generated {} channels, {} frames in {:?}",
        num_channels, num_frames, duration
    );
    if num_frames > 0 {
        let sample_rate = SAMPLE_RATE as u32;
        let spec = hound::WavSpec {
            channels: num_channels as u16,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&args.output, spec)?;

        if num_channels == 1 {
            let samples = waveform.flatten_all()?.to_vec1::<f32>()?;
            for &sample in samples.iter() {
                writer.write_sample(sample.clamp(-1.0_f32, 1.0_f32))?;
            }
        } else {
            let left_samples = waveform.get(0)?.to_vec1::<f32>()?;
            let right_samples = waveform.get(1)?.to_vec1::<f32>()?;
            for i in 0..num_frames {
                writer.write_sample(left_samples[i].clamp(-1.0_f32, 1.0_f32))?;
                writer.write_sample(right_samples[i].clamp(-1.0_f32, 1.0_f32))?;
            }
        }
        println!("Saved audio to {}", args.output);
    } else {
        println!("Warning: No audio generated.");
    }

    Ok(())
}
