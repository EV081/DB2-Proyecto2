import type { Metadata } from "next";
import { IBM_Plex_Mono, Plus_Jakarta_Sans } from "next/font/google";
import Link from "next/link";
import { SystemStatus } from "@/components/SystemStatus";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  variable: "--font-jakarta",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "GRID — Busqueda Multimodal",
  description: "Compara SPIMI (propio) vs pgvector/GIN/GiST en busqueda musical y de moda, en tiempo real.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="es"
      className={`${jakarta.variable} ${plexMono.variable} h-full antialiased`}
    >
      <head>
        <link rel="preconnect" href="https://api.fontshare.com" />
        <link
          href="https://api.fontshare.com/v2/css?f[]=cabinet-grotesk@500,700,900&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="flex min-h-full flex-col bg-background text-foreground">
        <header className="border-b border-border">
          <nav className="mx-auto flex max-w-[1680px] items-center gap-8 px-6 py-5 text-sm font-medium sm:px-10">
            <Link href="/" className="font-display text-xl font-black tracking-tight">
              GRID<span className="text-engine-spimi">.</span>
            </Link>
            <Link href="/music" className="text-muted hover:text-foreground">
              Musica
            </Link>
            <Link href="/fashion" className="text-muted hover:text-foreground">
              Fashion
            </Link>
            <span className="ml-auto">
              <SystemStatus />
            </span>
          </nav>
        </header>
        <main className="w-full flex-1 px-6 py-10 sm:px-10">{children}</main>
      </body>
    </html>
  );
}
