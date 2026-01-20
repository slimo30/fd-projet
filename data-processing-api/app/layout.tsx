import type { Metadata } from "next";
import type React from "react";
import { Geist, Geist_Mono, Inter } from "next/font/google";
import { Toaster } from "sonner";
import { NavigationHeader } from "@/components/navigation-header";
import { DataProvider } from "@/context/DataContext";
import "./globals.css";

// Load fonts
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const inter = Inter({ subsets: ["latin"] });

// Metadata
export const metadata: Metadata = {
  title: "Clustering Analysis Tool",
  description: "A tool for data clustering analysis",
};

// Root Layout
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.className} ${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <DataProvider>
          <NavigationHeader />
          <div className="min-h-[calc(100vh-4rem)]">
            {children}
          </div>
          <Toaster position="bottom-right" richColors />
        </DataProvider>
      </body>
    </html>
  );
}
