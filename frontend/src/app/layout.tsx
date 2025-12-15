import "~/styles/globals.css";

import { type Metadata } from "next";
import { Geist, Poppins } from "next/font/google";

export const metadata: Metadata = {
  title: "Sentiment Analysis",
  description: "Sentiment Analysis - Inside Out Edition",
  icons: [{ rel: "icon", url: "/favicon.png" }],
};

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

const poppins = Poppins({
  weight: ["400", "500", "600", "700", "800"],
  subsets: ["latin"],
  variable: "--font-poppins",
});

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geist.variable} ${poppins.variable} scroll-smooth`}>
      <body>{children}</body>
    </html>
  );
}
