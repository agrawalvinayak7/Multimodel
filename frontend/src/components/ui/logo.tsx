import React from 'react';

export const Logo = ({ className = "" }: { className?: string }) => {
    return (
        <svg
            width="40"
            height="40"
            viewBox="0 0 100 100"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className={className}
        >
            <defs>
                <radialGradient
                    id="joyGradient"
                    cx="0"
                    cy="0"
                    r="1"
                    gradientUnits="userSpaceOnUse"
                    gradientTransform="translate(30 30) rotate(90) scale(40)"
                >
                    <stop stopColor="var(--emotion-joy)" />
                    <stop offset="1" stopColor="#F59E0B" />
                </radialGradient>
                <radialGradient
                    id="sadnessGradient"
                    cx="0"
                    cy="0"
                    r="1"
                    gradientUnits="userSpaceOnUse"
                    gradientTransform="translate(70 70) rotate(90) scale(40)"
                >
                    <stop stopColor="var(--emotion-sadness)" />
                    <stop offset="1" stopColor="#2980B9" />
                </radialGradient>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="5" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>

            {/* Abstract intersection of emotions */}
            <circle cx="40" cy="40" r="30" fill="url(#joyGradient)" style={{ mixBlendMode: 'screen' }} opacity="0.9">
                <animate attributeName="cy" values="40;38;40" dur="4s" repeatCount="indefinite" />
            </circle>
            <circle cx="60" cy="60" r="30" fill="url(#sadnessGradient)" style={{ mixBlendMode: 'screen' }} opacity="0.9">
                <animate attributeName="cy" values="60;62;60" dur="5s" repeatCount="indefinite" />
            </circle>

            {/* Simple geometric representation of a "thought" or "spark" */}
            <path
                d="M50 20 L55 35 L70 40 L55 45 L50 60 L45 45 L30 40 L45 35 Z"
                fill="white"
                opacity="0.8"
                filter="url(#glow)"
                style={{ transformOrigin: '50px 40px' }}
            >
                <animateTransform
                    attributeName="transform"
                    type="scale"
                    values="1;0.8;1"
                    dur="3s"
                    repeatCount="indefinite"
                />
            </path>
        </svg>
    );
};
