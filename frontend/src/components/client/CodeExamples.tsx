"use client";

import { useState } from "react";
import { FiCode, FiTerminal } from "react-icons/fi";

function CodeExamples() {
    const [activeTab, setActiveTab] = useState<"ts" | "curl">("ts");

    const tsCode = `const fileType = "mp4";
    // 1. Get upload URL
    const res = await fetch("/api/upload-url", {
        method: "POST",
        headers: {
            Authorization: "Bearer " + apiKey,
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ fileType: fileType }),
    });

    if (!res.ok) {
        const error = await res.json();
        throw new Error(error?.error || "Failed to get upload URL");
    }

    const { url, fileId, key } = await res.json();

  // 2. Upload file to S3
    const uploadRes = await fetch(url, {
        method: "PUT",
        headers: { "Content-Type": file.type },
        body: file,
    });

    if (!uploadRes.ok) {
        throw new Error("Failed to upload file");
    }

    setStatus("analyzing");

  // 3. Analyze video
    const analysisRes = await fetch("/api/sentiment-inference", {
        method: "POST",
        headers:{
            Authorization: "Bearer " + apiKey,
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ key }),
    });

    if (!analysisRes.ok) {
        const error = await analysisRes.json();
        throw new Error(error?.error || "Failed to analyze video");
    }

    const analysis = await analysisRes.json();

    console.log("Analysis: ", analysis);`;

    const curlCode = `# 1. Get upload URL
    curl -X POST \\
    -H "Authorization: Bearer \${YOUR_API_KEY}" \\
    -H "Content-Type: application/json" \\
    -d '{"fileType": ".mp4"}' \\
    /api/upload-url

    # Response contains url and key

    # 2. Upload file to S3
    curl -X PUT \\
    -H "Content-Type: video/mp4" \\
    --data-binary @video.mp4 \\
    "\${url}"

    # 3. Analyze video
    curl -X POST \\
    -H "Authorization: Bearer \${YOUR_API_KEY}" \\
    -H "Content-Type: application/json" \\
    -d '{"key": "\${key}"}' \\
    /api/sentiment-inference`;

    return (
        <div className="rounded-lg bg-white border border-[#cccccc] p-6">
            <div className="mb-4 flex items-center justify-between">
                <div>
                    <h2 className="text-base font-semibold text-[#1d1d1d]">API Usage</h2>
                </div>
                <div className="flex gap-1 rounded bg-[#f5f5f5] p-1">
                    <button
                        onClick={() => setActiveTab("ts")}
                        className={`flex items-center gap-2 rounded px-3 py-1.5 text-xs font-semibold transition-colors ${activeTab === "ts"
                            ? "bg-[#5b7fc4] text-white"
                            : "text-[#666666] hover:text-[#1d1d1d]"
                            }`}
                    >
                        <FiCode />
                        TypeScript
                    </button>
                    <button
                        onClick={() => setActiveTab("curl")}
                        className={`flex items-center gap-2 rounded px-3 py-1.5 text-xs font-semibold transition-colors ${activeTab === "curl"
                            ? "bg-[#5b7fc4] text-white"
                            : "text-[#666666] hover:text-[#1d1d1d]"
                            }`}
                    >
                        <FiTerminal />
                        cURL
                    </button>
                </div>
            </div>

            <div className="overflow-hidden rounded border border-[#cccccc] bg-[#1d1d1d]">
                <div className="p-4">
                    <pre className="max-h-[300px] overflow-y-auto font-mono text-xs leading-relaxed text-[#e0e0e0]">
                        <code>
                            {activeTab === "ts" ? tsCode : curlCode}
                        </code>
                    </pre>
                </div>
            </div>
        </div>
    );
}

export default CodeExamples;
