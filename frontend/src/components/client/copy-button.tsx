"use client"

import { useState } from "react";
import { FiCheck, FiCopy } from "react-icons/fi";

function CopyButton({ text }: { text: string }) {
    const [copied, setCopied] = useState(false)

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(text)
            setCopied(true)
            setTimeout(() => setCopied(false), 2000)
        } catch {
            console.error("Failed to copy text to clipboard")
        }
    }
    return (
        <button
            className={`flex h-8 items-center justify-center gap-2 rounded px-3 py-1 text-xs font-semibold transition-colors ${copied
                ? "bg-[#e8f5e9] text-[#388e3c] border border-[#388e3c]"
                : "bg-[#5b7fc4] text-white hover:bg-[#4a6ba3] border-none"
                }`}
            onClick={handleCopy}>
            {copied ? (<FiCheck className="h-3.5 w-3.5" />) : (<FiCopy className="h-3.5 w-3.5" />)}
            {copied ? "Copied!" : "Copy"}
        </button>
    )
}

export default CopyButton