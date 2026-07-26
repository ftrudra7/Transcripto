import React from 'react';
import { GlassCard, GlassCardHeader, GlassCardTitle, GlassCardContent } from '@/components/ui/GlassCard';
import { Button } from '@/components/ui/Button';
import { Mic, FileText, UploadCloud, Sparkles } from 'lucide-react';

export function Home() {
  return (
    <div className="flex flex-col items-center">
      
      {/* Hero */}
      <div className="text-center py-16 flex flex-col items-center">
        <div className="inline-block px-4 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-sm font-semibold mb-6 uppercase tracking-wider">
          Enterprise Audio Intelligence
        </div>
        <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-b from-white to-indigo-200 bg-clip-text text-transparent mb-6 tracking-tight max-w-4xl">
          Intelligence at the Speed of Sound
        </h1>
        <p className="text-lg md:text-xl text-gray-400 max-w-2xl mb-12">
          Experience the world's most advanced transcription and language processing platform. Zero latency. Perfect accuracy.
        </p>

        {/* Control Dock */}
        <GlassCard className="w-full max-w-3xl flex flex-col items-center p-2 rounded-[2rem]">
          <div className="text-sm text-gray-400 mb-4 mt-2 font-medium">System Ready. Select an input method.</div>
          <div className="flex flex-wrap gap-4 justify-center pb-4">
            <Button size="lg" className="gap-2 text-base rounded-full">
              <Mic className="w-5 h-5" /> Record Audio
            </Button>
            <Button variant="outline" size="lg" className="gap-2 text-base rounded-full">
              <UploadCloud className="w-5 h-5" /> Upload File
            </Button>
            <Button variant="outline" size="lg" className="gap-2 text-base rounded-full">
              <Sparkles className="w-5 h-5" /> Live Sign AI
            </Button>
          </div>
        </GlassCard>
      </div>

      {/* Workspaces */}
      <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
        <GlassCard className="min-h-[400px] flex flex-col">
          <GlassCardHeader>
            <GlassCardTitle><FileText className="w-5 h-5 text-indigo-400" /> Transcription</GlassCardTitle>
          </GlassCardHeader>
          <GlassCardContent className="flex-1 flex items-center justify-center text-gray-500 italic">
            Audio transcription will stream here...
          </GlassCardContent>
        </GlassCard>

        <GlassCard className="min-h-[400px] flex flex-col">
          <GlassCardHeader>
            <GlassCardTitle><Sparkles className="w-5 h-5 text-purple-400" /> AI Summary</GlassCardTitle>
          </GlassCardHeader>
          <GlassCardContent className="flex-1 flex items-center justify-center text-gray-500 italic">
            Synthesized intelligence will appear here...
          </GlassCardContent>
        </GlassCard>
      </div>
      
    </div>
  );
}
