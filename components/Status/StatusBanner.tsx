import React from 'react';

type SystemStatus = 'operational' | 'degraded' | 'outage';

interface StatusBannerProps {
  status: SystemStatus;
  lastUpdated?: string;
}

const statusConfig = {
  operational: {
    label: 'All Systems Operational',
    color: 'text-green-500',
    borderColor: 'border-green-500/20',
    bgColor: 'bg-green-500/5',
    pulseColor: 'bg-green-500',
    icon: (
      <svg className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    ),
  },
  degraded: {
    label: 'Degraded Performance',
    color: 'text-yellow-500',
    borderColor: 'border-yellow-500/20',
    bgColor: 'bg-yellow-500/5',
    pulseColor: 'bg-yellow-500',
    icon: (
      <svg className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
  },
  outage: {
    label: 'Major System Outage',
    color: 'text-red-500',
    borderColor: 'border-red-500/20',
    bgColor: 'bg-red-500/5',
    pulseColor: 'bg-red-500',
    icon: (
      <svg className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
};

export default function StatusBanner({ status, lastUpdated }: StatusBannerProps) {
  const config = statusConfig[status];

  return (
    <div className={`relative overflow-hidden rounded-2xl border ${config.borderColor} ${config.bgColor} p-8 shadow-2xl backdrop-blur-xl transition-all duration-500`}>
      <div className={`absolute -right-20 -top-20 h-64 w-64 rounded-full ${config.pulseColor} opacity-5 blur-[100px] pointer-events-none`} />

      <div className="flex flex-col items-center justify-between gap-6 md:flex-row">
        <div className="flex items-center gap-6">
          <div className="relative flex h-16 w-16 items-center justify-center rounded-full bg-black/20 shadow-inner">
            <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${config.pulseColor} opacity-20 duration-1000`} />
            <div className={`${config.color}`}>{config.icon}</div>
          </div>

          <div className="flex flex-col">
            <h2 className={`text-2xl font-bold tracking-tight ${config.color} font-inter-display`}>
              {config.label}
            </h2>
            <p className="text-sm text-gray-400 font-satoshi">
              {lastUpdated ? `Last updated: ${lastUpdated}` : 'Real-time monitoring active'}
            </p>
          </div>
        </div>

        <div className="hidden h-12 w-px bg-white/5 md:block" />

        <div className="flex items-center gap-8">
            <div className="text-center">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Uptime (24h)</p>
                <p className="text-xl font-bold text-white font-mono">99.99%</p>
            </div>
             <div className="text-center">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Latency</p>
                <p className="text-xl font-bold text-white font-mono">24ms</p>
            </div>
        </div>
      </div>
    </div>
  );
}
