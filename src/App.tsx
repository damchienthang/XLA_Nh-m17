import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ActiveTab, TopBar } from './components/TopBar';
import { WorkbenchView } from './components/WorkbenchView';
import { BenchmarkRow, BenchmarkView } from './components/BenchmarkView';
import { DiagnosticsView } from './components/DiagnosticsView';
import { DATASET_SPECIMENS, DatasetSpecimen } from './forensics/dataset';
import { ColormapName } from './forensics/colormaps';
import { ForensicResult, analyzeImagePixels } from './forensics/analyzer';
import { decodeImageUrlToPixels } from './forensics/browserRunner';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('workbench');
  const [selectedSpecimen, setSelectedSpecimen] = useState<DatasetSpecimen>(
    DATASET_SPECIMENS.find((s) => s.id === 'degraded_04') ?? DATASET_SPECIMENS[0]
  );
  const [customImage, setCustomImage] = useState<{
    filename: string;
    url: string;
    width: number;
    height: number;
  } | null>(null);

  const [threshold, setThreshold] = useState<number>(0.5);
  const [colormap, setColormap] = useState<ColormapName>('JET');
  const [gain, setGain] = useState<number>(1.0);

  const [result, setResult] = useState<ForensicResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [benchmarkRows, setBenchmarkRows] = useState<BenchmarkRow[]>([]);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const runAnalysisForUrl = useCallback(async (url: string) => {
    setIsAnalyzing(true);
    setErrorMsg(null);
    try {
      const { rgba, width, height } = await decodeImageUrlToPixels(url);
      const res = analyzeImagePixels(rgba, width, height);
      setResult(res);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Lỗi phân tích ảnh');
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  useEffect(() => {
    const activeUrl = customImage ? customImage.url : selectedSpecimen.url;
    runAnalysisForUrl(activeUrl);
  }, [selectedSpecimen, customImage, runAnalysisForUrl]);

  const handleSelectSpecimen = (spec: DatasetSpecimen) => {
    if (customImage) {
      URL.revokeObjectURL(customImage.url);
      setCustomImage(null);
    }
    setSelectedSpecimen(spec);
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const objectUrl = URL.createObjectURL(file);
    try {
      const { width, height } = await decodeImageUrlToPixels(objectUrl);
      if (customImage) {
        URL.revokeObjectURL(customImage.url);
      }
      setCustomImage({
        filename: file.name,
        url: objectUrl,
        width,
        height,
      });
      setActiveTab('workbench');
    } catch {
      URL.revokeObjectURL(objectUrl);
      setErrorMsg('Không thể đọc tệp ảnh vừa tải lên. Vui lòng chọn PNG, JPG hoặc WEBP hợp lệ.');
    } finally {
      e.target.value = '';
    }
  };

  const handleInspectFromBenchmark = (spec: DatasetSpecimen) => {
    handleSelectSpecimen(spec);
    setActiveTab('workbench');
  };

  return (
    <div className="min-h-screen bg-[#070B14] text-slate-100">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        onChange={handleFileChange}
        className="hidden"
        aria-label="Tải tệp ảnh lên để phân tích"
      />

      <TopBar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onUploadClick={handleUploadClick}
      />

      {activeTab === 'workbench' && (
        <WorkbenchView
          selectedSpecimen={selectedSpecimen}
          customImage={customImage}
          onSelectSpecimen={handleSelectSpecimen}
          onTriggerUpload={handleUploadClick}
          threshold={threshold}
          onThresholdChange={setThreshold}
          colormap={colormap}
          onColormapChange={setColormap}
          gain={gain}
          onGainChange={setGain}
          result={result}
          isAnalyzing={isAnalyzing}
          errorMsg={errorMsg}
          onRunAnalysis={() => {
            const url = customImage ? customImage.url : selectedSpecimen.url;
            runAnalysisForUrl(url);
          }}
        />
      )}

      {activeTab === 'benchmark' && (
        <BenchmarkView
          rows={benchmarkRows}
          onBenchmarkComplete={setBenchmarkRows}
          threshold={threshold}
          onInspectRow={handleInspectFromBenchmark}
        />
      )}

      {activeTab === 'diagnostics' && <DiagnosticsView />}
    </div>
  );
}
