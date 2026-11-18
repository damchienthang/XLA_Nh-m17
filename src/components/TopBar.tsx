import React from 'react';
import { Upload } from 'lucide-react';

export type ActiveTab = 'workbench' | 'benchmark' | 'diagnostics';

interface TopBarProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  onUploadClick: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({ activeTab, onTabChange, onUploadClick }) => {
  const navItems: { id: ActiveTab; label: string }[] = [
    { id: 'workbench', label: 'Bàn Khám Nghiệm' },
    { id: 'benchmark', label: 'Benchmark 29 Ảnh' },
    { id: 'diagnostics', label: 'Chẩn Đoán Pipeline' },
  ];

  return (
    <header className="sticky top-0 z-30 border-b border-slate-800/90 bg-[#070B14]/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3 sm:px-6 md:flex-nowrap md:gap-8 md:py-3.5">
        <a
          href="#workbench"
          onClick={(e) => {
            e.preventDefault();
            onTabChange('workbench');
          }}
          className="text-sm font-semibold tracking-tight text-slate-100 whitespace-nowrap shrink-0 sm:text-base"
        >
          CO-SPY Forensic Studio
        </a>

        <nav
          aria-label="Điều hướng chính"
          className="order-3 flex w-full items-center gap-5 overflow-x-auto border-t border-slate-800/80 pt-2 md:order-none md:w-auto md:gap-6 md:border-t-0 md:pt-0"
        >
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onTabChange(item.id)}
                className={`relative py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 rounded sm:text-sm ${
                  isActive
                    ? 'text-cyan-400'
                    : 'text-slate-400 hover:text-slate-100'
                }`}
                aria-current={isActive ? 'page' : undefined}
              >
                {item.label}
                {isActive && (
                  <span
                    aria-hidden="true"
                    className="absolute right-0 bottom-0 left-0 h-0.5 bg-cyan-400"
                  />
                )}
              </button>
            );
          })}
        </nav>

        <div className="shrink-0">
          <button
            type="button"
            onClick={onUploadClick}
            className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-500 px-3 py-1.5 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-400 whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 sm:gap-2 sm:px-4 sm:py-2"
          >
            <Upload className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Tải ảnh kiểm tra</span>
          </button>
        </div>
      </div>
    </header>
  );
};
