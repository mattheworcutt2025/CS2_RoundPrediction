import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CS2 Round Predictor",
  description: "Predict CS2 round winners using deep learning (96.71% accuracy)",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="text-gray-100 min-h-screen relative">{children}</body>
    </html>
  );
}
