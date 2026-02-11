import React from 'react';

interface DayStatus {
  date: string;
  status: 'operational' | 'degraded' | 'outage' | 'maintenance';
  uptime: number;
}

interface UptimeGridProps {
  days?: DayStatus[];
}

const getStatusColor = (status: string) => {
  switch (status) {
    case 'operational': return 'bg-green-500 hover:bg-green-400 hover:shadow-[0_0_8px_rgba(74,222,128,0.5)]';
    case 'degraded': return 'bg-yellow-500 hover:bg-yellow-400 hover:shadow-[0_0_8px_rgba(250,204,21,0.5)]';
    case 'outage': return 'bg-red-500 hover:bg-red-400 hover:shadow-[0_0_8px_rgba(248,113,113,0.5)]';
    case 'maintenance': return 'bg-blue-500 hover:bg-blue-400 hover:shadow-[0_0_8px_rgba(96,165,250,0.5)]';
    default: return 'bg-white/5 hover:bg-white/10';
  }
};

const generateMockData = (): DayStatus[] => {
  const days: DayStatus[] = [];
  const today = new Date();
  // Generate 91 days (13 weeks * 7) to make the grid perfect
  for (let i = 90; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);

    let status: DayStatus['status'] = 'operational';
    let uptime = 100;

    const rand = Math.random();
    if (rand > 0.98) {
        status = 'outage';
        uptime = 92.5;
    } else if (rand > 0.95) {
        status = 'degraded';
        uptime = 98.2;
    }

    days.push({
      date: date.toISOString().split('T')[0],
      status,
      uptime,
    });
  }
  return days;
};

export default function UptimeGrid({ days = generateMockData() }: UptimeGridProps) {
  // Split into columns of 7 days (weeks)
  const weeks: DayStatus[][] = [];
  for (let i = 0; i < days.length; i += 7) {
    weeks.push(days.slice(i, i + 7));
  }

  return (
    <div className="w-full overflow-x-auto rounded-2xl border border-white/5 bg-black/40 p-6 backdrop-blur-xl">
      <div className="mb-6 flex items-center justify-between min-w-[600px]">
        <h3 className="font-inter-display text-lg font-semibold text-white">System Uptime History (90 Days)</h3>
        <div className="flex items-center gap-4 text-xs text-gray-400">
            <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-sm bg-green-500" /> Operational
            </div>
            <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-sm bg-yellow-500" /> Degraded
            </div>
            <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-sm bg-red-500" /> Outage
            </div>
        </div>
      </div>

      <div className="flex gap-1 min-w-[600px]">
        {weeks.map((week, weekIndex) => (
          <div key={weekIndex} className="flex flex-col gap-1">
            {week.map((day) => (
              <div
                key={day.date}
                className={`group relative h-3 w-3 rounded-[2px] ${getStatusColor(day.status)} transition-all duration-300`}
              >
                {/* Tooltip */}
                <div className="absolute bottom-full left-1/2 mb-2 hidden -translate-x-1/2 flex-col items-center whitespace-nowrap rounded-lg bg-gray-900 px-3 py-2 text-xs shadow-xl group-hover:flex z-50 border border-white/10">
                  <span className="font-semibold text-white">{day.date}</span>
                  <span className={`mt-1 font-mono ${day.status === 'operational' ? 'text-green-400' : day.status === 'degraded' ? 'text-yellow-400' : 'text-red-400'}`}>
                    {day.status.toUpperCase()}
                  </span>
                  <span className="text-gray-500">Uptime: {day.uptime}%</span>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="mt-4 flex justify-between text-xs text-gray-500 font-satoshi min-w-[600px]">
        <span>3 Months Ago</span>
        <span>Today</span>
      </div>
    </div>
  );
}
