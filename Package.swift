// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "AuraFaceKit",
    platforms: [
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "AuraFaceKit",
            targets: ["AuraFaceKit"]
        )
    ],
    targets: [
        .target(
            name: "AuraFaceKit"
        )
    ]
)
