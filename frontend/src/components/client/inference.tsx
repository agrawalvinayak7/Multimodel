"use client";

import { useState } from "react";
import UploadVideo from "./UploadVideo";

const EMOTION_EMOJI: Record<string, string> = {
    anger: "😡",
    disgust: "🤢",
    fear: "😨",
    joy: "😄",
    neutral: "😐",
    sadness: "😢",
    surprise: "😲",
};

const EMOTION_COLORS: Record<string, string> = {
    anger: "var(--emotion-anger)",
    disgust: "var(--emotion-disgust)",
    fear: "var(--emotion-fear)",
    joy: "var(--emotion-joy)",
    neutral: "var(--emotion-neutral)",
    sadness: "var(--emotion-sadness)",
    surprise: "#FF9F1C", // Orange-ish for surprise
};

const SENTIMENT_EMOJI: Record<string, string> = {
    negative: "😡",
    neutral: "😐",
    positive: "😄",
};

const SENTIMENT_COLORS: Record<string, string> = {
    negative: "var(--emotion-anger)",
    neutral: "var(--emotion-sadness)",
    positive: "var(--emotion-joy)",
};

interface InferenceProps {
    quota: {
        secretKey: string;
    };
}

export type Analysis = {
    analysis: {
        utterances: Array<{
            start_time: number;
            end_time: number;
            text: string;
            emotions: Array<{ label: string; confidence: number }>;
            sentiments: Array<{ label: string; confidence: number }>;
        }>;
    };
};

export function Inference({ quota }: InferenceProps) {
    const [analysis, setAnalysis] = useState<Analysis | null>();

    const getAverageScores = () => {
        if (!analysis?.analysis.utterances.length) return null;

        // Aggregate all the scores
        const emotionScores: Record<string, number[]> = {};
        const sentimentScores: Record<string, number[]> = {};

        analysis.analysis.utterances.forEach((utterance) => {
            utterance.emotions.forEach((emotion) => {
                emotionScores[emotion.label] ??= [];
                emotionScores[emotion.label]!.push(emotion.confidence);
            });
            utterance.sentiments.forEach((sentiment) => {
                sentimentScores[sentiment.label] ??= [];
                sentimentScores[sentiment.label]!.push(sentiment.confidence);
            });
        });

        // Calculate the average
        const avgEmotions = Object.entries(emotionScores).map(
            ([label, scores]) => ({
                label,
                confidence: scores.reduce((a, b) => a + b, 0) / scores.length,
            })
        );

        const avgSentiments = Object.entries(sentimentScores).map(
            ([label, scores]) => ({
                label,
                confidence: scores.reduce((a, b) => a + b, 0) / scores.length,
            })
        );

        // Sort by confidence, get the top score
        const topEmotion = avgEmotions.sort(
            (a, b) => b.confidence - a.confidence
        )[0];
        const topSentiment = avgSentiments.sort(
            (a, b) => b.confidence - a.confidence
        )[0];

        return { topEmotion, topSentiment };
    };

    const averages = getAverageScores();

    return (
        <div className="flex h-fit w-full flex-col gap-6">
            {/* Header */}
            <div className="border-b border-[#e0e0e0] pb-4">
                <h2 className="text-xl font-semibold text-[#1d1d1d]">Inference</h2>
            </div>

            <UploadVideo onAnalysis={setAnalysis} apiKey={quota.secretKey} />

            <div className="mt-4">
                <h2 className="mb-4 text-base font-semibold text-[#1d1d1d]">Overall Analysis</h2>

                {averages ? (
                    <div className="rounded-lg bg-white border border-[#cccccc] flex flex-wrap items-center justify-around gap-8 p-8 animate-fade-in">

                        {/* Primary Emotion */}
                        <div className="flex flex-col items-center">
                            <span className="mb-2 text-xs font-semibold text-[#666666] uppercase">Emotion</span>
                            <div className="relative flex h-24 w-24 items-center justify-center rounded-full bg-[#f5f5f5] border border-[#cccccc]">
                                <span className="text-6xl">
                                    {averages.topEmotion?.label
                                        ? EMOTION_EMOJI[averages.topEmotion.label]
                                        : ""}
                                </span>
                            </div>
                            <span className="mt-3 text-xl font-bold text-[#1d1d1d] capitalize">
                                {averages.topEmotion?.label}
                            </span>
                            <span className="text-sm text-[#666666] mt-1">
                                {averages.topEmotion?.confidence.toFixed(3)} confidence
                            </span>
                        </div>

                        {/* Primary Sentiment */}
                        <div className="flex flex-col items-center">
                            <span className="mb-2 text-xs font-semibold text-[#666666] uppercase">Sentiment</span>
                            <div className="relative flex h-24 w-24 items-center justify-center rounded-full bg-[#f5f5f5] border border-[#cccccc]">
                                <span className="text-6xl">
                                    {averages.topSentiment?.label
                                        ? SENTIMENT_EMOJI[averages.topSentiment.label]
                                        : ""}
                                </span>
                            </div>
                            <span className="mt-3 text-xl font-bold text-[#1d1d1d] capitalize">
                                {averages.topSentiment?.label}
                            </span>
                            <span className="text-sm text-[#666666] mt-1">
                                {averages.topSentiment?.confidence.toFixed(3)} confidence
                            </span>
                        </div>
                    </div>
                ) : (
                    <div className="rounded-lg bg-white border-2 border-dashed border-[#cccccc] flex h-32 w-full items-center justify-center">
                        <p className="text-[#999999] italic">Upload a video to see overall analysis</p>
                    </div>
                )}
            </div>

            <div className="mt-4">
                <h2 className="mb-4 text-base font-semibold text-[#1d1d1d]">Analysis of utterances</h2>

                {analysis ? (
                    <div className="flex flex-col gap-4">
                        {analysis?.analysis.utterances.map((utterance, index) => {
                            return (
                                <div
                                    key={
                                        utterance.start_time.toString() +
                                        utterance.end_time.toString()
                                    }
                                    className="rounded-lg bg-white border border-[#cccccc] flex w-full flex-col gap-6 md:flex-row md:gap-8 p-6"
                                >
                                    {/* Time and text */}
                                    <div className="flex w-full md:max-w-[200px] flex-col">
                                        <div className="inline-flex w-fit items-center rounded bg-[#e3f2fd] px-2 py-1 text-xs font-mono text-[#1d1d1d]">
                                            {Number(utterance.start_time).toFixed(1)}s -{" "}
                                            {Number(utterance.end_time).toFixed(1)}s
                                        </div>
                                        <div className="mt-3 text-sm text-[#1d1d1d] leading-relaxed">
                                            &quot;{utterance.text}&quot;
                                        </div>
                                    </div>

                                    {/* Emotions */}
                                    <div className="flex w-full flex-1 flex-col gap-3">
                                        <span className="text-xs font-semibold text-[#666666] uppercase">Emotions</span>
                                        <div className="space-y-2">
                                            {utterance.emotions.map((emo) => (
                                                <div key={emo.label} className="flex items-center gap-3">
                                                    <span className="text-lg w-6">{EMOTION_EMOJI[emo.label]}</span>
                                                    <div className="flex-1 h-1.5 rounded-full bg-[#e0e0e0] overflow-hidden">
                                                        <div
                                                            className="h-full rounded-full transition-all duration-500"
                                                            style={{
                                                                width: `${emo.confidence * 100}%`,
                                                                backgroundColor: EMOTION_COLORS[emo.label] ?? '#ccc'
                                                            }}
                                                        />
                                                    </div>
                                                    <span className="text-xs text-[#666666] w-8 text-right">{(emo.confidence * 100).toFixed(0)}%</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Sentiments */}
                                    <div className="flex w-full flex-1 flex-col gap-3">
                                        <span className="text-xs font-semibold text-[#666666] uppercase">Sentiments</span>
                                        <div className="space-y-2">
                                            {utterance.sentiments.map((sentiment) => (
                                                <div key={sentiment.label} className="flex items-center gap-3">
                                                    <span className="text-lg w-6">{SENTIMENT_EMOJI[sentiment.label]}</span>
                                                    <div className="flex-1 h-1.5 rounded-full bg-[#e0e0e0] overflow-hidden">
                                                        <div
                                                            className="h-full rounded-full transition-all duration-500"
                                                            style={{
                                                                width: `${sentiment.confidence * 100}%`,
                                                                backgroundColor: SENTIMENT_COLORS[sentiment.label] ?? '#ccc'
                                                            }}
                                                        />
                                                    </div>
                                                    <span className="text-xs text-[#666666] w-8 text-right">{(sentiment.confidence * 100).toFixed(0)}%</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    <div className="rounded-lg bg-white border-2 border-dashed border-[#cccccc] flex h-24 w-full items-center justify-center">
                        <p className="text-[#999999] text-sm">Upload a video to analyze details</p>
                    </div>
                )}
            </div>
        </div>
    );
}