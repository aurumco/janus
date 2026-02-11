import React from 'react';
import type { Metadata } from 'next';
import Layout from '@/components/Layout';
import StatusBanner from '@/components/Status/StatusBanner';
import ServiceList, { Service } from '@/components/Status/ServiceList';
import UptimeGrid from '@/components/Status/UptimeGrid';

export const metadata: Metadata = {
  title: 'System Status | Janus',
  description: 'Real-time operational status of Janus services.',
};

const mockServices: Service[] = [
  { id: '1', name: 'API Gateway', status: 'operational', uptime: '99.99%', description: 'Main entry point for all client requests' },
  { id: '2', name: 'Authentication Service', status: 'operational', uptime: '100%', description: 'User identity and access management' },
  { id: '3', name: 'Market Data Feed', status: 'degraded', uptime: '98.50%', description: 'Real-time websocket connections to exchanges' },
  { id: '4', name: 'Execution Engine', status: 'operational', uptime: '99.95%', description: 'Order matching and routing system' },
  { id: '5', name: 'Database Cluster', status: 'operational', uptime: '99.99%', description: 'Primary distributed storage' },
  { id: '6', name: 'Analytics Pipeline', status: 'maintenance', uptime: '99.00%', description: 'Batch processing for historical data' },
];

// Determine global status
const getGlobalStatus = (services: Service[]) => {
  if (services.some(s => s.status === 'outage')) return 'outage';
  if (services.some(s => s.status === 'degraded')) return 'degraded';
  return 'operational';
};

export default function StatusPage() {
  const globalStatus = getGlobalStatus(mockServices);
  const lastUpdated = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <Layout className="space-y-12">
      {/* Header Section */}
      <div className="space-y-4 text-center sm:text-left">
        <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl font-inter-display">
          System Status
        </h1>
        <p className="max-w-2xl text-lg text-gray-400 font-satoshi">
          Current status of Janus platform services and real-time incident updates.
        </p>
      </div>

      {/* Global Status Banner */}
      <StatusBanner status={globalStatus} lastUpdated={lastUpdated} />

      {/* Service List */}
      <div className="space-y-6">
        <div className="flex items-center gap-3">
            <h2 className="text-2xl font-semibold text-white font-inter-display">Service Health</h2>
            <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-xs font-medium text-gray-400">
                {mockServices.length} Systems
            </span>
        </div>
        <ServiceList services={mockServices} />
      </div>

      {/* Uptime Grid */}
      <div className="space-y-6">
         <h2 className="text-2xl font-semibold text-white font-inter-display">Historical Uptime</h2>
         <UptimeGrid />
      </div>

      {/* Incident History Placeholder (Bonus) */}
      <div className="rounded-2xl border border-white/5 bg-white/5 p-8 backdrop-blur-sm">
        <h3 className="text-xl font-semibold text-white mb-6 font-inter-display">Recent Incidents</h3>
        <div className="space-y-8">
            <div className="relative pl-8 border-l border-white/10">
                <span className="absolute -left-1.5 top-1.5 h-3 w-3 rounded-full bg-yellow-500 ring-4 ring-black" />
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <h4 className="font-medium text-white text-lg">Market Data Latency</h4>
                    <span className="text-sm text-gray-500 font-mono">Oct 24, 14:30 UTC</span>
                </div>
                <p className="text-sm text-gray-400 mt-2 leading-relaxed">
                    We are investigating reports of increased latency in the ETH-USDT data feed. Engineering team is scaling up the ingestion nodes.
                </p>
            </div>
             <div className="relative pl-8 border-l border-white/10">
                <span className="absolute -left-1.5 top-1.5 h-3 w-3 rounded-full bg-green-500 ring-4 ring-black" />
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <h4 className="font-medium text-white text-lg">Scheduled Maintenance Completed</h4>
                     <span className="text-sm text-gray-500 font-mono">Oct 12, 09:00 UTC</span>
                </div>
                <p className="text-sm text-gray-400 mt-2 leading-relaxed">
                    Database migration completed successfully. All services are back online with improved query performance.
                </p>
            </div>
        </div>
      </div>
    </Layout>
  );
}
