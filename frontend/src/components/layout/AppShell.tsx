import React from 'react';
import { Outlet } from 'react-router-dom';
import { AnimatedBackground } from '../ui/AnimatedBackground';
import { BrainCircuit } from 'lucide-react';

export function AppShell() {
  return (
    <div className="flex flex-col min-h-screen relative z-[1]">
      <AnimatedBackground />
      
      <nav className="sticky top-4 z-50 mx-auto w-max max-w-[calc(100%-2rem)] px-6 py-3 flex items-center gap-8 rounded-full bg-slate-900/60 border border-white/10 backdrop-blur-xl shadow-2xl">
        <a href="/" className="font-heading font-extrabold text-xl flex items-center gap-2 bg-gradient-to-br from-white to-indigo-300 bg-clip-text text-transparent">
          <BrainCircuit className="w-6 h-6 text-indigo-400" />
          Transcripto
        </a>
        <ul className="flex gap-6 list-none">
          <li><a href="#" className="text-gray-400 hover:text-white transition-colors font-medium text-sm">Platform</a></li>
          <li><a href="#sign" className="text-gray-400 hover:text-white transition-colors font-medium text-sm">Sign AI</a></li>
          <li><a href="#history" className="text-gray-400 hover:text-white transition-colors font-medium text-sm">History</a></li>
        </ul>
      </nav>

      <main className="flex-1 w-full max-w-7xl mx-auto px-6 py-12">
        <Outlet />
      </main>
    </div>
  );
}
