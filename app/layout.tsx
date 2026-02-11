import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Header from '@/components/Header';

// Mocking local font imports as requested
// import localFont from 'next/font/local';
// const satoshi = localFont({ src: './fonts/Satoshi-Variable.woff2', variable: '--font-satoshi' });
// const interDisplay = localFont({ src: './fonts/InterDisplay-Medium.woff2', variable: '--font-inter-display' });

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Janus System Status',
  description: 'Real-time operational status of the Janus platform.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} min-h-screen bg-black text-white antialiased`}>
        <div className="flex flex-col min-h-screen">
          <Header />
          <main className="flex-grow flex flex-col">
            {children}
          </main>
          <footer className="py-8 border-t border-white/10 text-center text-sm text-gray-500">
            © {new Date().getFullYear()} Janus Platform. All rights reserved.
          </footer>
        </div>
      </body>
    </html>
  );
}
