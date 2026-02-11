import Link from 'next/link';

export default function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/10 bg-black/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-4">
          <Link href="/" className="text-xl font-bold tracking-tight text-white hover:text-gray-300 transition-colors">
            Janus
          </Link>
          <span className="hidden sm:inline-block rounded-full bg-white/10 px-2 py-0.5 text-xs font-medium text-gray-400">
            v5.0.0
          </span>
        </div>
        <nav className="flex gap-6">
          <Link href="/" className="text-sm font-medium text-gray-400 hover:text-white transition-colors">
            Dashboard
          </Link>
          <Link href="/status" className="text-sm font-medium text-white transition-colors">
            Status
          </Link>
          <a href="#" className="text-sm font-medium text-gray-400 hover:text-white transition-colors">
            Docs
          </a>
        </nav>
      </div>
    </header>
  );
}
