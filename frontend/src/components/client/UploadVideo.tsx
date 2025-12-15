"use client";

import { useState } from "react";
import { FiUpload, FiFilm, FiActivity, FiCheckCircle } from "react-icons/fi";
import type { Analysis } from "./inference";

interface UploadVideoProps {
  apiKey: string;
  onAnalysis: (analysis: Analysis) => void;
}

function UploadVideo({ apiKey, onAnalysis }: UploadVideoProps) {
  const [status, setStatus] = useState<"idle" | "uploading" | "analyzing" | "success">("idle");
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    try {
      setStatus("uploading");
      setError(null);

      const fileType = `.${file.name.split(".").pop()}`;

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
        const errorData = await res.json() as { error?: string };
        throw new Error(errorData?.error ?? "Failed to get upload URL");
      }

      const responseData = await res.json() as { url: string; key: string };
      const { url, key } = responseData;

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
        headers: {
          Authorization: "Bearer " + apiKey,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ key }),
      });

      if (!analysisRes.ok) {
        const errorData = await analysisRes.json() as { error?: string };
        throw new Error(errorData?.error ?? "Failed to analyze video");
      }

      const analysis = await analysisRes.json() as Analysis;

      console.log("Analysis: ", analysis);
      onAnalysis(analysis);
      setStatus("success");

      // Reset back to idle after a short delay to allow uploading another video
      setTimeout(() => setStatus("idle"), 3000);

    } catch (error) {
      setError(error instanceof Error ? error.message : "Upload failed");
      console.error("Upload failed", error);
      setStatus("idle");
    }
  };

  return (
    <div className="flex w-full flex-col gap-4">
      <div className="relative overflow-hidden rounded-lg border-2 border-dashed border-[#999999] bg-white p-10 transition-all hover:border-[#5b7fc4]">
        <input
          type="file"
          accept="video/mp4,video/mov,video/avi"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void handleUpload(file);
          }}
          id="video-upload"
          disabled={status !== "idle"}
        />
        <label
          htmlFor="video-upload"
          className={`flex cursor-pointer flex-col items-center justify-center gap-4 ${status !== "idle" ? "pointer-events-none opacity-50" : ""}`}
        >
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-[#5b7fc4]">
            <FiUpload className="h-8 w-8 text-white" />
          </div>

          <div className="text-center">
            <h3 className="text-base font-semibold text-[#1d1d1d]">
              Upload a video
            </h3>
            <p className="mt-2 text-sm text-[#666666]">
              Get started with sentiment detection...
            </p>
            <p className="mt-1 text-xs text-[#999999]">
              MP4, MOV, AVI (Max 1 min)
            </p>
          </div>
        </label>

        {/* Loading / Status Overlay */}
        {status !== "idle" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/95 z-10">
            {status === "uploading" && (
              <>
                <div className="h-12 w-12 text-[#5b7fc4]">
                  <FiFilm className="h-full w-full" />
                </div>
                <p className="mt-4 text-base font-semibold text-[#1d1d1d]">Uploading...</p>
              </>
            )}
            {status === "analyzing" && (
              <>
                <div className="h-12 w-12 text-[#5b7fc4]">
                  <FiActivity className="h-full w-full" />
                </div>
                <p className="mt-4 text-base font-semibold text-[#1d1d1d]">Analyzing...</p>
              </>
            )}
            {status === "success" && (
              <>
                <div className="h-12 w-12 text-[#388e3c]">
                  <FiCheckCircle className="h-full w-full" />
                </div>
                <p className="mt-4 text-base font-semibold text-[#1d1d1d]">Analysis Complete!</p>
              </>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="rounded border border-[#d32f2f] bg-[#ffebee] p-3 text-center text-sm text-[#d32f2f]">
          {error}
        </div>
      )}
    </div>
  );
}

export default UploadVideo;