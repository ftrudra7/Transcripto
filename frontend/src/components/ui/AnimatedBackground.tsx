import React from 'react';

export function AnimatedBackground() {
  return (
    <div className="fixed inset-0 z-[-2] bg-[var(--color-background)] overflow-hidden">
      <div className="mesh-bg"></div>
      <div className="noise-bg"></div>
    </div>
  );
}
