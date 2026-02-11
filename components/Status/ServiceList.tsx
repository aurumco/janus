import React from 'react';

type ServiceStatus = 'operational' | 'degraded' | 'outage' | 'maintenance';

export interface Service {
  id: string;
  name: string;
  status: ServiceStatus;
  description?: string;
  uptime: string;
}

interface ServiceListProps {
  services: Service[];
}

const statusColorMap = {
  operational: 'bg-green-500',
  degraded: 'bg-yellow-500',
  outage: 'bg-red-500',
  maintenance: 'bg-blue-500',
};

const statusTextMap = {
  operational: 'Operational',
  degraded: 'Degraded',
  outage: 'Outage',
  maintenance: 'Maintenance',
};

export default function ServiceList({ services }: ServiceListProps) {
  return (
    <div className="grid gap-4 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
      {services.map((service) => (
        <div
          key={service.id}
          className="group relative flex flex-col justify-between overflow-hidden rounded-xl border border-white/5 bg-white/5 p-6 backdrop-blur-sm transition-all duration-300 hover:border-white/10 hover:bg-white/10"
        >
          <div className="flex items-start justify-between">
            <h3 className="font-inter-display text-lg font-semibold text-white">
              {service.name}
            </h3>
            <div className={`flex items-center gap-2 rounded-full px-2.5 py-0.5 text-xs font-medium border border-white/5 bg-black/20`}>
              <span className={`h-2 w-2 rounded-full ${statusColorMap[service.status]} shadow-[0_0_8px_rgba(0,0,0,0.5)]`} />
              <span className="capitalize text-gray-300">{statusTextMap[service.status]}</span>
            </div>
          </div>

          <p className="mt-2 text-sm text-gray-400 font-satoshi">
            {service.description || 'Core system component'}
          </p>

          <div className="mt-6 flex items-center justify-between border-t border-white/5 pt-4">
            <span className="text-xs font-medium text-gray-500">90-Day Uptime</span>
            <span className="font-mono text-sm font-bold text-green-400">{service.uptime}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
