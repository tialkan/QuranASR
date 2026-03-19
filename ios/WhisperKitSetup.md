# WhisperKit iOS Integration Guide

Fine-tuned Quran ASR model integration with WhisperKit from scratch.

## Prerequisites

- Xcode 15+
- iOS 16+ deployment target
- CoreML models from `convert/export_coreml.py`

---

## 1. Add WhisperKit via Swift Package Manager

In Xcode: **File → Add Package Dependencies**

```
https://github.com/argmaxinc/WhisperKit
```

Select version: `0.9.0` or latest stable.

---

## 2. Bundle the CoreML Models

1. Create a folder `QuranASRModel/` in your Xcode project
2. Copy these files into it:
   - `AudioEncoder.mlmodelc`
   - `MelSpectrogram.mlmodelc`
   - `TextDecoder.mlmodelc`
   - `config.json`
3. In Xcode: **Add files to project** → select the folder → ensure **"Copy items if needed"** is checked
4. Set **Target Membership** for all files

---

## 3. Swift Integration

### QuranASR.swift

```swift
import WhisperKit
import Foundation

/// Manages Quran speech recognition using a fine-tuned WhisperKit model.
class QuranASR {

    private var whisperKit: WhisperKit?

    // Path to bundled CoreML model folder
    private let modelFolder: String = {
        guard let path = Bundle.main.path(forResource: "QuranASRModel", ofType: nil) else {
            fatalError("QuranASRModel folder not found in bundle")
        }
        return path
    }()

    /// Initialize WhisperKit with the custom Quran model.
    func setup() async throws {
        let config = WhisperKitConfig(
            modelFolder: modelFolder,
            verbose: false,
            logLevel: .error,
            // Force Arabic language — skip language detection
            language: "ar",
            task: .transcribe
        )
        whisperKit = try await WhisperKit(config)
    }

    /// Transcribe an audio chunk and return the normalized (harakah-stripped) text.
    /// - Parameter audioURL: URL to an audio file (WAV or MP3, 16kHz recommended)
    /// - Returns: Harakah-stripped Arabic text for database matching
    func transcribe(audioURL: URL) async throws -> TranscriptionResult {
        guard let kit = whisperKit else {
            throw ASRError.notInitialized
        }

        let results = try await kit.transcribe(audioPath: audioURL.path)

        guard let firstResult = results.first else {
            return TranscriptionResult(harekeli: "", harekesiz: "", segments: [])
        }

        // Model output is already harekesiz — matches DB directly
        let text = firstResult.text.trimmingCharacters(in: .whitespacesAndNewlines)

        let segments = firstResult.segments.map { seg in
            TranscriptionSegment(
                text: seg.text.trimmingCharacters(in: .whitespacesAndNewlines),
                start: seg.start,
                end: seg.end
            )
        }

        return TranscriptionResult(text: text, segments: segments)
    }

    // MARK: - Hareke Stripping

    /// Remove Arabic diacritics (tashkeel) from text.
    /// Unicode range: U+064B–U+065F (standard tashkeel) + extended marks
    func removeHarekeler(_ text: String) -> String {
        let diacriticRanges: [ClosedRange<UInt32>] = [
            0x0610...0x061A,  // Extended Arabic marks
            0x064B...0x065F,  // Standard tashkeel (fathah, kasrah, dammah, etc.)
            0x06D6...0x06DC,  // Quranic annotation signs
            0x06DF...0x06E8,  // More annotation signs
            0x06EA...0x06ED,  // More annotation signs
        ]

        return text.unicodeScalars
            .filter { scalar in
                !diacriticRanges.contains { $0.contains(scalar.value) }
            }
            .map { Character($0) }
            .reduce("") { $0 + String($1) }
    }
}

// MARK: - Supporting Types

struct TranscriptionResult {
    let harekeli: String   // Raw model output (with diacritics)
    let harekesiz: String  // Stripped text (for DB matching)
    let segments: [TranscriptionSegment]
}

struct TranscriptionSegment {
    let text: String
    let start: Float  // seconds
    let end: Float    // seconds
}

enum ASRError: LocalizedError {
    case notInitialized
    case audioLoadFailed(URL)

    var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "WhisperKit not initialized. Call setup() first."
        case .audioLoadFailed(let url):
            return "Failed to load audio from: \(url.path)"
        }
    }
}
```

### VerseMatcherService.swift

```swift
import Foundation

/// Matches transcribed text against the Quran verse database.
class VerseMatcherService {

    /// Find the best matching verse for the given transcribed text.
    /// - Parameter harekesiz: Diacritic-stripped Arabic text from ASR
    /// - Returns: Matched verse info, or nil if no confident match found
    func findVerse(matching harekesiz: String) -> VerseMatch? {
        let normalized = normalize(harekesiz)
        guard !normalized.isEmpty else { return nil }

        // Strategy 1: Exact substring match against all verses
        if let exact = QuranDatabase.shared.exactMatch(normalized) {
            return VerseMatch(surahId: exact.surahId, verseId: exact.verseId,
                              confidence: 1.0, matchType: .exact)
        }

        // Strategy 2: Fuzzy/token-based match (Jaccard similarity on words)
        let inputWords = Set(normalized.split(separator: " ").map(String.init))
        var bestScore = 0.0
        var bestVerse: (surahId: Int, verseId: Int)?

        QuranDatabase.shared.forEachVerse { surahId, verseId, text in
            let verseWords = Set(normalize(text).split(separator: " ").map(String.init))
            let intersection = inputWords.intersection(verseWords).count
            let union = inputWords.union(verseWords).count
            let jaccard = union > 0 ? Double(intersection) / Double(union) : 0.0

            if jaccard > bestScore {
                bestScore = jaccard
                bestVerse = (surahId, verseId)
            }
        }

        guard bestScore > 0.5, let verse = bestVerse else { return nil }

        return VerseMatch(surahId: verse.surahId, verseId: verse.verseId,
                          confidence: bestScore, matchType: .fuzzy)
    }

    private func normalize(_ text: String) -> String {
        // Collapse whitespace, trim
        return text
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}

struct VerseMatch {
    let surahId: Int
    let verseId: Int
    let confidence: Double
    let matchType: MatchType

    enum MatchType { case exact, fuzzy }

    var description: String {
        "Surah \(surahId):\(verseId) (confidence: \(String(format: "%.2f", confidence * 100))%, \(matchType))"
    }
}
```

### Usage in ViewController / ViewModel

```swift
// AppDelegate or scene setup:
let quranASR = QuranASR()
let verseMatcher = VerseMatcherService()

Task {
    try await quranASR.setup()
}

// When audio chunk is ready:
func processAudioChunk(url: URL) async {
    do {
        let result = try await quranASR.transcribe(audioURL: url)
        print("Transcribed: \(result.harekesiz)")

        if let match = verseMatcher.findVerse(matching: result.harekesiz) {
            print("Matched: \(match.description)")
            // Navigate to surah match.surahId, verse match.verseId
        }
    } catch {
        print("ASR error: \(error)")
    }
}
```

---

## 4. Audio Chunking (AVFoundation)

For live microphone input, chunk audio into 3–5 second segments:

```swift
import AVFoundation

class AudioChunker {
    private let engine = AVAudioEngine()
    private let chunkDuration: TimeInterval = 4.0  // seconds
    private var buffer: AVAudioPCMBuffer?
    private var chunkStartTime = Date()

    var onChunkReady: ((URL) -> Void)?

    func start() throws {
        let inputNode = engine.inputNode
        let format = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, time in
            self?.handleBuffer(buffer)
        }

        try engine.start()
    }

    private func handleBuffer(_ buffer: AVAudioPCMBuffer) {
        // Accumulate buffer, flush every chunkDuration seconds
        let elapsed = Date().timeIntervalSince(chunkStartTime)
        if elapsed >= chunkDuration {
            flushChunk()
            chunkStartTime = Date()
        }
    }

    private func flushChunk() {
        // Write accumulated audio to temp WAV file and call onChunkReady
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".wav")
        // ... write buffer to url ...
        onChunkReady?(url)
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}
```

---

## 5. Model File Size Reference

| File | Size (approx) |
|------|---------------|
| `AudioEncoder.mlmodelc` | ~90 MB |
| `TextDecoder.mlmodelc` | ~130 MB |
| `MelSpectrogram.mlmodelc` | ~1 MB |
| Total | ~221 MB |

> **Note:** Use App Store on-demand resources (ODR) to avoid embedding 221MB in the initial download if needed.

---

## 6. Testing Checklist

- [ ] Model loads without crash on device
- [ ] 3-second Arabic audio chunk transcribes correctly
- [ ] Hareke stripping works (output matches DB format)
- [ ] Verse matching returns correct surah:verse for 10 test samples
- [ ] WER < 5% on held-out reciter (not in training set)
- [ ] Latency < 2s per chunk on iPhone 12+

---

## References

- WhisperKit: https://github.com/argmaxinc/WhisperKit
- whisperkittools: https://github.com/argmaxinc/whisperkittools
- CoreML conversion guide: https://huggingface.co/docs/transformers/main/en/model_doc/whisper
