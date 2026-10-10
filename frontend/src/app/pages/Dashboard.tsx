import { useState, useEffect, useRef } from "react";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import {
  Sparkles,
  Leaf,
  MessageCircle,
  Copy,
  User,
  Calendar,
  FileText,
  AlertTriangle,
  TrendingUp,
  CheckCircle,
  Download,
  Volume2,
  Loader2,
} from "lucide-react";
import { Navbar } from "../components/Navbar";
import { Link } from "react-router";
import { toast } from "sonner";
import { Toaster } from "../components/ui/sonner";

// Matches backend LabParameter schema exactly
type LabParameter = {
  name: string;
  value: number;
  unit: string;
  reference_low: number;
  reference_high: number;
  status: "High" | "Low" | "Normal";
};

// Fallback mock data matching backend LabParameter shape
const defaultLabData: LabParameter[] = [
  { name: "Hemoglobin", value: 14.2, unit: "g/dL", reference_low: 13.0, reference_high: 17.0, status: "Normal" },
  { name: "White Blood Cell Count", value: 11.5, unit: "×10³/μL", reference_low: 4.0, reference_high: 10.0, status: "High" },
  { name: "Platelet Count", value: 220, unit: "×10³/μL", reference_low: 150, reference_high: 400, status: "Normal" },
  { name: "Blood Glucose (Fasting)", value: 118, unit: "mg/dL", reference_low: 70, reference_high: 100, status: "High" },
  { name: "Total Cholesterol", value: 245, unit: "mg/dL", reference_low: 0, reference_high: 200, status: "High" },
  { name: "Vitamin D", value: 22, unit: "ng/mL", reference_low: 30, reference_high: 100, status: "Low" },
];

const API_BASE = ""; // Proxy handles /api

export function Dashboard() {
  const [labData, setLabData] = useState<LabParameter[]>(defaultLabData);
  const [patientName, setPatientName] = useState("John Anderson");
  const [reportDate, setReportDate] = useState("Feb 20, 2026");
  const [summary, setSummary] = useState("");
  const [intellectualAudio, setIntellectualAudio] = useState("");
  const [preventiveGuidance, setPreventiveGuidance] = useState("");
  const [doctorQuestions, setDoctorQuestions] = useState<string[]>([]);
  const [hasTrendData, setHasTrendData] = useState(false);
  const [filter, setFilter] = useState<"all" | "abnormal" | "normal">("all");
  const [isCopied, setIsCopied] = useState(false);
  const [fromApi, setFromApi] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

  // Audio state
  const [language, setLanguage] = useState("English");
  const [isAudioLoading, setIsAudioLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const reportRef = useRef<HTMLDivElement>(null);

  // Load API response (AnalysisResponse) from localStorage set by Upload page
  useEffect(() => {
    const storedLang = localStorage.getItem("selectedLanguage");
    if (storedLang) setLanguage(storedLang);

    const stored = localStorage.getItem("labReportData");
    if (stored) {
      try {
        const data = JSON.parse(stored);
        if (data.patient_name) setPatientName(data.patient_name);
        if (data.report_date) setReportDate(data.report_date);
        if (Array.isArray(data.parameters) && data.parameters.length > 0) {
          setLabData(data.parameters);
        }
        if (data.summary) setSummary(data.summary);
        if (data.intellectual_audio) setIntellectualAudio(data.intellectual_audio);
        if (data.preventive_guidance) setPreventiveGuidance(data.preventive_guidance);
        if (Array.isArray(data.doctor_questions) && data.doctor_questions.length > 0) {
          setDoctorQuestions(data.doctor_questions);
        }
        setFromApi(true);
      } catch {
        // Fall back to mock data silently
      }
    }

    // Check if trend data is available
    const trend = localStorage.getItem("trendData");
    if (trend) setHasTrendData(true);
  }, []);

  const abnormalCount = labData.filter((item) => item.status !== "Normal").length;

  const sortedData = [...labData].sort((a, b) => {
    const aAbnormal = a.status !== "Normal" ? 0 : 1;
    const bAbnormal = b.status !== "Normal" ? 0 : 1;
    return aAbnormal - bAbnormal;
  });

  const filteredData = sortedData.filter((item) => {
    if (filter === "abnormal") return item.status !== "Normal";
    if (filter === "normal") return item.status === "Normal";
    return true;
  });

  const handleCopyQuestions = () => {
    const text = doctorQuestions.map((q: string, i: number) => `${i + 1}. ${q}`).join("\n");
    navigator.clipboard.writeText(text);
    setIsCopied(true);
    toast.success("Questions copied to clipboard!");
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleDownloadPDF = () => {
    // We use the browser's native print functionality which seamlessly handles oklab/oklch colors
    // and layout formatting better than html2canvas for modern Tailwind CSS.
    window.print();
  };

  const handlePlaySummaryAudio = async () => {
    const AudioScript = intellectualAudio || summary; // Fallback for old data
    if (!AudioScript) return;

    // Toggle playback if already loaded and playing
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        setIsPlaying(false);
        return;
      }
    }

    setIsAudioLoading(true);
    try {
      // Create fresh audio object if none exists or we want to replay
      const response = await fetch(`${API_BASE}/api/v1/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: AudioScript.replace(/[\*\#\[\]\\]/g, " "), // Remove remaining raw markdown symbols if any slip through
          language: language,
        }),
      });

      if (!response.ok) throw new Error("TTS request failed");

      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);

      audio.onended = () => setIsPlaying(false);
      audio.onerror = () => {
        setIsPlaying(false);
        toast.error("Error playing audio stream.");
      };

      audioRef.current = audio;
      audio.play();
      setIsPlaying(true);
    } catch (error) {
      console.error("Audio generation error:", error);
      toast.error("Failed to generate audio summary.");
    } finally {
      setIsAudioLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8fafc]">
      <style>
        {`
          @media print {
            .no-print {
              display: none !important;
            }
            body {
              background-color: #fff !important;
            }
            /* Ensure report container takes full width and no margins */
            .print-container {
              max-width: 100% !important;
              margin: 0 !important;
              padding: 0 !important;
            }
          }
        `}
      </style>
      <Toaster />
      <Navbar />

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8 flex items-start justify-between flex-wrap gap-4 print:hidden">
          <div>
            <h1 className="text-3xl font-bold text-[#0f172a] mb-2">
              Your Health Snapshot
            </h1>
            <p className="text-[#64748b]">
              {fromApi
                ? "AI analysis of your uploaded lab report"
                : "Comprehensive analysis of your latest health report"}
            </p>
          </div>
          <div className="flex gap-3 items-center">
            {hasTrendData && (
              <Link
                to="/trends"
                className="inline-flex items-center gap-2 bg-gradient-to-r from-[#0d9488] to-[#7c3aed] text-white px-5 py-2.5 rounded-lg font-semibold text-sm hover:shadow-lg transition-all hover:scale-105"
              >
                <TrendingUp className="w-4 h-4" />
                View Trend Analysis
              </Link>
            )}
            <button
              onClick={handleDownloadPDF}
              className="inline-flex items-center gap-2 bg-[#0f172a] text-white px-5 py-2.5 rounded-lg font-semibold text-sm hover:bg-[#1e293b] hover:shadow-lg transition-all hover:scale-105"
            >
              <Download className="w-4 h-4" />
              Download PDF
            </button>
          </div>
        </div>

        {/* Printable Report Container */}
        <div ref={reportRef} className="space-y-6">
          {/* Patient Information Card */}
          <div className="bg-white rounded-xl shadow-sm border-l-4 border-[#7c3aed] p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-[#0f172a]">Patient Information</h3>
              <span
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold ${abnormalCount > 0
                  ? "bg-[#fb923c]/10 text-[#fb923c]"
                  : "bg-[#22c55e]/10 text-[#22c55e]"
                  }`}
              >
                {abnormalCount > 0 ? (
                  <>
                    <AlertTriangle className="w-4 h-4" />
                    {abnormalCount}{" "}
                    {abnormalCount === 1 ? "value needs" : "values need"} attention
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-4 h-4" />
                    All values normal
                  </>
                )}
              </span>
            </div>

            <div className="grid md:grid-cols-3 gap-6">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-[#0d9488]/10 rounded-lg flex items-center justify-center flex-shrink-0">
                  <User className="w-5 h-5 text-[#0d9488]" />
                </div>
                <div>
                  <div className="text-xs text-[#64748b] mb-1">Patient Name</div>
                  <div className="font-medium text-[#0f172a]">{patientName}</div>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-[#0d9488]/10 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Calendar className="w-5 h-5 text-[#0d9488]" />
                </div>
                <div>
                  <div className="text-xs text-[#64748b] mb-1">Test Date</div>
                  <div className="font-medium text-[#0f172a]">{reportDate}</div>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-[#0d9488]/10 rounded-lg flex items-center justify-center flex-shrink-0">
                  <FileText className="w-5 h-5 text-[#0d9488]" />
                </div>
                <div>
                  <div className="text-xs text-[#64748b] mb-1">Parameters</div>
                  <div className="font-medium text-[#0f172a]">{labData.length} extracted</div>
                </div>
              </div>
            </div>
          </div>

          {/* Parameters Table */}
          <div className="bg-white rounded-xl shadow-sm border border-[#e2e8f0] mb-6 overflow-hidden">
            <div className="border-b border-[#e2e8f0] px-6 py-4 bg-[#f8fafc]">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-[#64748b]">Filter:</span>
                <button
                  onClick={() => setFilter("all")}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${filter === "all"
                    ? "bg-[#0d9488] text-white"
                    : "bg-white text-[#64748b] hover:bg-[#e2e8f0]"
                    }`}
                >
                  All ({labData.length})
                </button>
                <button
                  onClick={() => setFilter("abnormal")}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${filter === "abnormal"
                    ? "bg-[#ef4444] text-white"
                    : "bg-white text-[#64748b] hover:bg-[#e2e8f0]"
                    }`}
                >
                  Abnormal Only ({abnormalCount})
                </button>
                <button
                  onClick={() => setFilter("normal")}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${filter === "normal"
                    ? "bg-[#22c55e] text-white"
                    : "bg-white text-[#64748b] hover:bg-[#e2e8f0]"
                    }`}
                >
                  Normal Only ({labData.length - abnormalCount})
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-[#0d9488] text-white">
                    <th className="text-left px-6 py-4 font-bold">Parameter</th>
                    <th className="text-left px-6 py-4 font-bold">Value</th>
                    <th className="text-left px-6 py-4 font-bold">Reference Range</th>
                    <th className="text-left px-6 py-4 font-bold">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredData.map((item, index) => {
                    const isAbnormal = item.status !== "Normal";
                    const bgColor = isAbnormal ? "bg-[#ef4444]/10" : "bg-white";
                    const borderColor = isAbnormal ? "border-l-4 border-l-[#ef4444]" : "";
                    const badgeColor = isAbnormal ? "text-[#ef4444]" : "text-[#22c55e]";
                    const badgeBg = isAbnormal ? "bg-[#ef4444]/10" : "bg-[#22c55e]/10";

                    return (
                      <tr
                        key={index}
                        className={`border-b border-[#e2e8f0] last:border-b-0 ${bgColor} ${borderColor} hover:shadow-[inset_3px_0_0_0_#0d9488] transition-shadow`}
                      >
                        <td className="px-6 py-4 font-medium text-[#0f172a]">{item.name}</td>
                        <td className="px-6 py-4 text-[#0f172a] font-mono">
                          {item.value} {item.unit}
                        </td>
                        <td className="px-6 py-4 text-[#64748b]">
                          {item.reference_low} – {item.reference_high} {item.unit}
                        </td>
                        <td className="px-6 py-4">
                          <span
                            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${badgeColor} ${badgeBg}`}
                          >
                            {isAbnormal && <AlertTriangle className="w-3 h-3" />}
                            {item.status.toUpperCase()}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Two Column Layout */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* AI Health Summary */}
            <div className="bg-[#7c3aed]/5 rounded-xl border border-[#7c3aed]/20 overflow-hidden">
              <div className="bg-[#7c3aed] px-6 py-4 flex items-center justify-between">
                <h3 className="font-bold text-white">What Your Results Mean</h3>
                <div className="flex gap-2 items-center print:hidden">
                  <button
                    onClick={handlePlaySummaryAudio}
                    disabled={isAudioLoading || !summary}
                    className="inline-flex items-center gap-1.5 bg-white/20 hover:bg-white/30 backdrop-blur-sm px-3 py-1.5 rounded-full text-xs font-semibold text-white transition-all disabled:opacity-50"
                  >
                    {isAudioLoading ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Volume2 className={`w-3.5 h-3.5 ${isPlaying ? 'animate-pulse text-[#d8b4fe]' : ''}`} />
                    )}
                    {isPlaying ? "Stop Audio" : "Play Audio"}
                  </button>
                  <span className="inline-flex items-center gap-1 bg-white/20 backdrop-blur-sm px-3 py-1.5 rounded-full text-xs font-medium text-white">
                    <Sparkles className="w-3.5 h-3.5" />
                    AI Generated
                  </span>
                </div>
              </div>
              <div className="p-6">
                {summary ? (
                  <p className="text-[#0f172a] leading-relaxed whitespace-pre-line">{summary}</p>
                ) : (
                  <p className="text-[#64748b] italic leading-relaxed">
                    Upload a lab report to get an AI-generated summary of your results.
                  </p>
                )}
                <p className="text-xs text-[#64748b] italic border-t border-[#7c3aed]/20 pt-4 mt-4">
                  This is not medical advice. Always consult your doctor.
                </p>
              </div>
            </div>

            {/* Preventive Guidance */}
            <div className="bg-white rounded-xl shadow-sm border border-[#e2e8f0] overflow-hidden">
              <div className="bg-[#0d9488] px-6 py-4 flex items-center gap-2">
                <Leaf className="w-5 h-5 text-white" />
                <h3 className="font-bold text-white">What You Can Do</h3>
              </div>
              <div className="p-6">
                {preventiveGuidance ? (
                  <p className="text-[#0f172a] leading-relaxed whitespace-pre-line">
                    {preventiveGuidance}
                  </p>
                ) : (
                  <p className="text-[#64748b] italic leading-relaxed">
                    Upload a lab report to get personalized preventive guidance from AI.
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Doctor Questions */}
          <div className="bg-white rounded-xl shadow-sm border-l-4 border-[#7c3aed] p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <MessageCircle className="w-5 h-5 text-[#7c3aed]" />
                  <h3 className="font-bold text-[#0f172a]">
                    Ask Your Doctor These Questions
                  </h3>
                </div>
                <p className="text-xs text-[#64748b]">
                  {doctorQuestions.length > 0
                    ? "Generated by AI based on your lab results"
                    : "Upload a report to generate personalized questions"}
                </p>
              </div>
              {doctorQuestions.length > 0 && (
                <button
                  onClick={handleCopyQuestions}
                  className="flex items-center gap-2 px-4 py-2 bg-[#0d9488] text-white rounded-lg hover:bg-[#0d9488]/90 transition-colors text-sm font-medium"
                >
                  <Copy className="w-4 h-4" />
                  {isCopied ? "Copied ✓" : "Copy All Questions"}
                </button>
              )}
            </div>

            <div className="space-y-3">
              {doctorQuestions.length > 0 ? (
                doctorQuestions.map((question: string, index: number) => (
                  <div
                    key={index}
                    className="flex items-start gap-3 p-4 bg-[#f8fafc] rounded-lg border border-[#e2e8f0]"
                  >
                    <span className="flex items-center justify-center w-6 h-6 bg-[#7c3aed] text-white rounded-full text-xs font-bold flex-shrink-0">
                      {index + 1}
                    </span>
                    <span className="text-[#0f172a] leading-relaxed">{question}</span>
                  </div>
                ))
              ) : (
                <div className="text-center py-6 text-[#64748b] italic text-sm">
                  No doctor questions yet. Upload a lab report to generate personalized questions.
                </div>
              )}
            </div>
          </div>

          {/* Disclaimer Footer */}
          <div className="mt-10 pb-6 text-center">
            <p className="text-xs text-[#94a3b8]">
              This is not a medical diagnosis. Always consult your doctor.
            </p>
          </div>

        </div> {/* End of Printable Container */}
      </div>
    </div>
  );
}
