import os
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from collections import defaultdict
import subprocess
from tqdm import tqdm

# Load model and encoders (using absolute paths)
import os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = load_model(os.path.join(base_path, "music_genre_model.h5"))
scaler = joblib.load(os.path.join(base_path, "scaler.save"))
label_encoder = joblib.load(os.path.join(base_path, "label_encoder.save"))
genres = label_encoder.classes_

# Global variable to store progress updates
progress_updates = []


def remove_vocals_with_demucs(audio_path):
    """Removes vocals using Demucs CLI."""
    output_dir = os.path.expanduser("~/separated/htdemucs")
    os.makedirs(output_dir, exist_ok=True)

    print("→ Removing vocals using Demucs (this may take a minute)...")
    cmd = ["python", "-m", "demucs", "--two-stems=vocals", audio_path]
    subprocess.run(cmd, check=True)

    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    no_vocal_path = os.path.join(output_dir, song_name, "no_vocals.wav")

    if os.path.exists(no_vocal_path):
        print("✓ Vocals removed successfully.")
        return no_vocal_path
    else:
        raise FileNotFoundError(
            "Demucs output not found — check Demucs installation or paths."
        )


def extract_features(segment, sr=22050):
    """Extracts 58 consistent audio features."""
    features = [len(segment)]

    def mean_var(f):
        return float(np.mean(f)), float(np.var(f))

    chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
    rms = librosa.feature.rms(y=segment)
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=segment)
    harmony = librosa.effects.harmonic(y=segment)
    perceptr = librosa.effects.percussive(y=segment)
    tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
    tempo = float(np.mean(tempo)) if np.ndim(tempo) else float(tempo)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)

    for f in [chroma, rms, centroid, bandwidth, rolloff, zcr]:
        m, v = mean_var(f)
        features += [m, v]

    features += [
        float(np.mean(harmony)),
        float(np.var(harmony)),
        float(np.mean(perceptr)),
        float(np.var(perceptr)),
        tempo,
    ]

    for i in range(20):
        features.append(float(np.mean(mfcc[i])))
        features.append(float(np.var(mfcc[i])))

    return np.array(features).reshape(1, -1)


def predict_genre_song(audio_path, segment_sec=3, remove_vocals=False):
    """Predict genre for long song with confidence voting."""
    global progress_updates
    progress_updates = []  # Reset progress updates

    # Note: Vocal removal disabled for simplicity
    progress_updates.append(
        {
            "message": "📁 Loading audio file...",
            "progress": 5,
            "current_segment": 0,
            "total_segments": 0,
        }
    )

    # Convert to wav if mp3
    if audio_path.lower().endswith(".mp3"):
        progress_updates.append(
            {
                "message": "🔄 Converting MP3 to WAV...",
                "progress": 10,
                "current_segment": 0,
                "total_segments": 0,
            }
        )
        wav_path = audio_path.replace(".mp3", ".wav")
        AudioSegment.from_mp3(audio_path).export(wav_path, format="wav")
        audio_path = wav_path

    progress_updates.append(
        {
            "message": "🎵 Loading audio data...",
            "progress": 15,
            "current_segment": 0,
            "total_segments": 0,
        }
    )

    # Load audio and calculate total segments
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    segment_len = sr * segment_sec
    total_len = len(y)
    total_segments_calculated = max(1, int(total_len // segment_len))

    genre_confidence = defaultdict(float)
    total_segments = 0

    duration_str = f"{total_len / sr:.1f}s"
    progress_updates.append(
        {
            "message": f"⏱️ Analyzing {duration_str} audio in {segment_sec}s segments...",
            "progress": 20,
            "current_segment": 0,
            "total_segments": total_segments_calculated,
        }
    )

    print(f"\nAnalyzing {total_len / sr:.1f}s audio in {segment_sec}s chunks...\n")

    for start in tqdm(range(0, total_len, segment_len)):
        segment = y[start : start + segment_len]
        if len(segment) < sr:  # skip too short
            continue

        segment_num = total_segments + 1

        # Calculate progress percentage: 20% base + (segment_num / total_segments_calculated) * 70%
        progress_percent = 20 + int((segment_num / total_segments_calculated) * 60)
        progress_updates.append(
            {
                "message": f"⚙️ Processing segment {segment_num}/{total_segments_calculated}...",
                "progress": min(progress_percent, 80),
                "current_segment": segment_num,
                "total_segments": total_segments_calculated,
            }
        )

        features = extract_features(segment, sr)
        features_scaled = scaler.transform(features)
        preds = model.predict(features_scaled, verbose=0)[0]
        conf = np.max(preds)
        genre_idx = np.argmax(preds)
        genre = label_encoder.inverse_transform([genre_idx])[0]

        genre_confidence[genre] += conf
        total_segments += 1

    if not total_segments:
        progress_updates.append(
            {
                "message": "❌ No valid segments found.",
                "progress": 0,
                "current_segment": 0,
                "total_segments": 0,
            }
        )
        print("❌ No valid segments found.")
        return None

    progress_updates.append(
        {
            "message": "📊 Calculating confidence scores...",
            "progress": 90,
            "current_segment": total_segments_calculated,
            "total_segments": total_segments_calculated,
        }
    )

    # Normalize confidence scores
    for genre in genre_confidence:
        genre_confidence[genre] /= total_segments

    final_genre = max(genre_confidence, key=genre_confidence.get)

    progress_updates.append(
        {
            "message": f"🎵 Predicted genre: {final_genre.upper()}",
            "progress": 95,
            "current_segment": total_segments_calculated,
            "total_segments": total_segments_calculated,
        }
    )

    progress_updates.append(
        {
            "message": f"✨ Processing complete!",
            "progress": 100,
            "current_segment": total_segments_calculated,
            "total_segments": total_segments_calculated,
        }
    )

    print("\nSegment-wise confidence summary:")
    for g, c in sorted(genre_confidence.items(), key=lambda x: -x[1]):
        print(f"  {g:<10}: {c:.3f}")

    print(f"\n🎧 Final Predicted Genre: {final_genre.upper()} 🎵")
    return final_genre, dict(genre_confidence)
