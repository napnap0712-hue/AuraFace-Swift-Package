import Foundation
import CoreML
import CryptoKit
import UIKit
import CoreVideo

public enum AuraFaceError: LocalizedError {
    case invalidModelURL
    case modelDownloadFailed(statusCode: Int)
    case modelChecksumMismatch
    case imageConversionFailed
    case embeddingMissing
    case embeddingDimensionMismatch(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidModelURL:
            return "AuraFaceモデルのURLが不正です。"
        case .modelDownloadFailed(let statusCode):
            return "AuraFaceモデルのダウンロードに失敗しました（HTTP \(statusCode)）。"
        case .modelChecksumMismatch:
            return "ダウンロードしたAuraFaceモデルのSHA-256が一致しません。"
        case .imageConversionFailed:
            return "顔画像をCore ML入力へ変換できませんでした。"
        case .embeddingMissing:
            return "AuraFaceのembedding出力を取得できませんでした。"
        case .embeddingDimensionMismatch(let count):
            return "AuraFaceのembedding次元が512ではありません（\(count)）。"
        }
    }
}

@MainActor
public final class AuraFaceManager {
    public static let shared = AuraFaceManager()

    public nonisolated static let inputSize = CGSize(width: 112, height: 112)
    public static let embeddingDimension = 512

    // RuiSumida/AuraFace-v1-CoreML
    private static let modelURLString =
        "https://huggingface.co/RuiSumida/AuraFace-v1-CoreML/resolve/main/FaceEmbedding.mlmodel?download=true"

    // Hugging Face掲載のFaceEmbedding.mlmodel SHA-256
    private static let expectedSHA256 =
        "9cb10bef2141a36619bb1fdbf1e0e14e2519c6da4b7e9b9969a4d67702d7122b"

    private var loadedModel: MLModel?

    private init() {}

    public var isPrepared: Bool {
        loadedModel != nil
    }

    /// 初回だけ約130MBのCore MLモデルをダウンロード・SHA-256検証・コンパイルします。
    /// 2回目以降は端末内に保存したコンパイル済みモデルを利用します。
    public func prepare() async throws {
        if loadedModel != nil { return }

        let paths = try modelPaths()
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine

        if FileManager.default.fileExists(atPath: paths.compiled.path) {
            loadedModel = try await MLModel.load(
                contentsOf: paths.compiled,
                configuration: configuration
            )
            return
        }

        if FileManager.default.fileExists(atPath: paths.source.path) {
            let digest = try Self.sha256(of: paths.source)
            if digest.lowercased() != Self.expectedSHA256 {
                try FileManager.default.removeItem(at: paths.source)
            }
        }

        if !FileManager.default.fileExists(atPath: paths.source.path) {
            try await downloadModel(to: paths.source)
        }

        let compiledTemporaryURL = try await MLModel.compileModel(at: paths.source)

        if FileManager.default.fileExists(atPath: paths.compiled.path) {
            try FileManager.default.removeItem(at: paths.compiled)
        }

        try FileManager.default.moveItem(
            at: compiledTemporaryURL,
            to: paths.compiled
        )

        loadedModel = try await MLModel.load(
            contentsOf: paths.compiled,
            configuration: configuration
        )
    }

    /// 112×112に切り出し済み・できれば5点アラインメント済みの顔画像を渡します。
    public func generateEmbedding(for image: UIImage) async throws -> [Float] {
        if loadedModel == nil {
            try await prepare()
        }

        guard let model = loadedModel else {
            throw AuraFaceError.embeddingMissing
        }

        let resized = image.resizedForAuraFace()
        guard let pixelBuffer = resized.auraFacePixelBuffer() else {
            throw AuraFaceError.imageConversionFailed
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "faceImage": MLFeatureValue(pixelBuffer: pixelBuffer)
        ])

        let output = try await model.prediction(from: provider)

        guard let multiArray = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw AuraFaceError.embeddingMissing
        }

        guard multiArray.count == Self.embeddingDimension else {
            throw AuraFaceError.embeddingDimensionMismatch(multiArray.count)
        }

        var values = [Float]()
        values.reserveCapacity(multiArray.count)

        for index in 0..<multiArray.count {
            values.append(multiArray[index].floatValue)
        }

        return Self.l2Normalize(values)
    }

    public func generateEmbeddings(for images: [UIImage]) async throws -> [[Float]] {
        var results = [[Float]]()
        results.reserveCapacity(images.count)

        for image in images {
            results.append(try await generateEmbedding(for: image))
        }

        return results
    }

    public nonisolated func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return -1 }

        var dot: Float = 0
        var normL: Float = 0
        var normR: Float = 0

        for index in lhs.indices {
            dot += lhs[index] * rhs[index]
            normL += lhs[index] * lhs[index]
            normR += rhs[index] * rhs[index]
        }

        let denominator = sqrt(normL) * sqrt(normR)
        return denominator > 0 ? dot / denominator : -1
    }

    public nonisolated func euclideanDistance(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count, !lhs.isEmpty else {
            return .greatestFiniteMagnitude
        }

        var sum: Float = 0
        for index in lhs.indices {
            let difference = lhs[index] - rhs[index]
            sum += difference * difference
        }

        return sqrt(sum)
    }

    /// 0.4は元サンプル互換用。園児写真の本番閾値として固定しないでください。
    public nonisolated func isSamePerson(
        _ lhs: [Float],
        _ rhs: [Float],
        threshold: Float = 0.4
    ) -> Bool {
        cosineSimilarity(lhs, rhs) >= threshold
    }

    private func modelPaths() throws -> (source: URL, compiled: URL) {
        let applicationSupport = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )

        let folder = applicationSupport
            .appendingPathComponent("AuraFaceKit", isDirectory: true)

        try FileManager.default.createDirectory(
            at: folder,
            withIntermediateDirectories: true
        )

        return (
            folder.appendingPathComponent("FaceEmbedding.mlmodel"),
            folder.appendingPathComponent("FaceEmbedding.mlmodelc", isDirectory: true)
        )
    }

    private func downloadModel(to destination: URL) async throws {
        guard let url = URL(string: Self.modelURLString) else {
            throw AuraFaceError.invalidModelURL
        }

        let (temporaryURL, response) = try await URLSession.shared.download(from: url)

        if let http = response as? HTTPURLResponse,
           !(200...299).contains(http.statusCode) {
            throw AuraFaceError.modelDownloadFailed(statusCode: http.statusCode)
        }

        let digest = try Self.sha256(of: temporaryURL)
        guard digest.lowercased() == Self.expectedSHA256 else {
            throw AuraFaceError.modelChecksumMismatch
        }

        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }

        try FileManager.default.moveItem(at: temporaryURL, to: destination)
    }

    private nonisolated static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()

        while true {
            let data = try handle.read(upToCount: 1_048_576) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }

        return hasher.finalize()
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private nonisolated static func l2Normalize(_ values: [Float]) -> [Float] {
        let norm = sqrt(values.reduce(Float(0)) { partial, value in
            partial + value * value
        })

        guard norm > 0 else { return values }
        return values.map { $0 / norm }
    }
}

private extension UIImage {
    func resizedForAuraFace() -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true

        return UIGraphicsImageRenderer(
            size: AuraFaceManager.inputSize,
            format: format
        ).image { _ in
            draw(in: CGRect(origin: .zero, size: AuraFaceManager.inputSize))
        }
    }

    func auraFacePixelBuffer() -> CVPixelBuffer? {
        let width = Int(AuraFaceManager.inputSize.width)
        let height = Int(AuraFaceManager.inputSize.height)

        var pixelBuffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue |
                        CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            return nil
        }

        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)

        guard let cgImage = self.cgImage else {
            return nil
        }

        context.draw(
            cgImage,
            in: CGRect(x: 0, y: 0, width: width, height: height)
        )

        return pixelBuffer
    }
}