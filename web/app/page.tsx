"use client";

import { useState, useCallback } from "react";
import { predict } from "./predict";

const MAPS = [
  "de_dust2", "de_mirage", "de_inferno", "de_nuke",
  "de_overpass", "de_vertigo", "de_ancient", "de_anubis",
];

interface GameState {
  ct_alive: number;
  t_alive: number;
  ct_health_avg: number;
  t_health_avg: number;
  ct_armor_pct: number;
  t_armor_pct: number;
  ct_rifles: number;
  t_rifles: number;
  ct_awps: number;
  t_awps: number;
  bomb_planted: boolean;
  time_remaining: number;
  round_num: number;
  ct_score: number;
  t_score: number;
  map: string;
  ct_money: number;
  t_money: number;
}

const defaultState: GameState = {
  ct_alive: 5,
  t_alive: 5,
  ct_health_avg: 100,
  t_health_avg: 100,
  ct_armor_pct: 100,
  t_armor_pct: 100,
  ct_rifles: 4,
  t_rifles: 4,
  ct_awps: 1,
  t_awps: 1,
  bomb_planted: false,
  time_remaining: 115,
  round_num: 15,
  ct_score: 7,
  t_score: 7,
  map: "de_dust2",
  ct_money: 20000,
  t_money: 20000,
};

function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  side,
  suffix = "",
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  side: "t" | "ct" | "neutral";
  suffix?: string;
}) {
  const valueColor =
    side === "t" ? "text-yellow-400" : side === "ct" ? "text-[#5d9bec]" : "text-white";
  const accentColor =
    side === "t" ? "accent-yellow-500" : side === "ct" ? "accent-blue-400" : "accent-gray-400";

  return (
    <div className="flex items-center gap-3">
      <label className="w-32 text-sm text-gray-300 shrink-0">{label}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`flex-1 h-2 rounded-lg appearance-none cursor-pointer ${accentColor}`}
      />
      <span className={`w-20 text-right font-mono text-sm font-semibold ${valueColor}`}>
        {value}{suffix}
      </span>
    </div>
  );
}

export default function Home() {
  const [state, setState] = useState<GameState>(defaultState);
  const [result, setResult] = useState<{ ct: number; t: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = useCallback(
    <K extends keyof GameState>(key: K, val: GameState[K]) => {
      setState((prev) => ({ ...prev, [key]: val }));
    },
    []
  );

  const runPrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      const prob = await predict(state);
      setResult({ ct: prob, t: 1 - prob });
    } catch (e) {
      console.error(e);
      setError(e instanceof Error ? e.message : "Model failed to load");
    }
    setLoading(false);
  };

  const ctWinPct = result ? Math.round(result.ct * 100) : null;
  const tWinPct = result ? Math.round(result.t * 100) : null;

  return (
    <main className="max-w-4xl mx-auto px-4 py-8 relative">
      {/* Header with T and CT */}
      <div className="relative mb-10">
        <div className="text-center">
          <div className="inline-block mb-4 px-4 py-1.5 rounded-full bg-[#0f3460]/50 border border-[#0f3460]">
            <span className="text-xs text-gray-300 tracking-wide">MGTA 611 &mdash; Course Project</span>
          </div>
          <h1 className="text-5xl font-black mb-3 tracking-tight">
            <span className="text-yellow-400">CS2</span>{" "}
            <span className="text-white">Round Predictor</span>
          </h1>
          <p className="text-gray-400 text-sm mb-3">
            Deep learning model with <span className="text-yellow-400 font-semibold">96.71% accuracy</span>
            <span className="text-gray-600"> | </span>
            104 features &middot; 1.2M parameters &middot; runs in your browser
          </p>
          <p className="text-gray-500 text-xs">
            By <span className="text-gray-300 font-medium">Ratul Sarker</span> &amp; <span className="text-gray-300 font-medium">Matthew Orcutt</span>
          </p>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-6 p-4 rounded-xl bg-red-900/30 border border-red-700 text-red-300 text-center">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="mb-8 rounded-2xl bg-[#16213e]/90 backdrop-blur-sm border border-[#0f3460]/50 overflow-hidden">
          {/* Winner Banner */}
          <div className={`py-4 px-6 text-center ${
            ctWinPct! > tWinPct!
              ? "bg-gradient-to-r from-[#1a3a6e] to-[#0f3460]"
              : ctWinPct! < tWinPct!
              ? "bg-gradient-to-r from-[#5c4a00] to-[#3d3100]"
              : "bg-[#1a1a2e]"
          }`}>
            <div className="text-xs uppercase tracking-widest text-gray-400 mb-1">Predicted Winner</div>
            <div className={`text-3xl font-black uppercase tracking-wide ${
              ctWinPct! > tWinPct! ? "text-[#5d9bec]" : ctWinPct! < tWinPct! ? "text-yellow-400" : "text-gray-300"
            }`}>
              {ctWinPct! > tWinPct! ? "Counter-Terrorists" : ctWinPct! < tWinPct! ? "Terrorists" : "Coin Flip"}
            </div>
            <div className="text-sm text-gray-400 mt-1">
              {ctWinPct! > 60 || tWinPct! > 60
                ? "Strong advantage"
                : ctWinPct! > 55 || tWinPct! > 55
                ? "Moderate advantage"
                : "Close call"}
            </div>
          </div>

          {/* Probability Bar */}
          <div className="p-6">
            <div className="flex items-center gap-4">
              <div className="flex-1 text-right">
                <div className="text-3xl font-bold text-yellow-400">{tWinPct}%</div>
                <div className="text-sm text-yellow-600 font-medium">T Side</div>
              </div>
              <div className="w-full max-w-md h-10 bg-[#0a0a1a] rounded-full overflow-hidden flex border border-gray-700/50">
                <div
                  className="h-full bg-gradient-to-r from-yellow-600 to-yellow-500 transition-all duration-500"
                  style={{ width: `${tWinPct}%` }}
                />
                <div
                  className="h-full bg-gradient-to-r from-[#4b7bec] to-[#3867d6] transition-all duration-500"
                  style={{ width: `${ctWinPct}%` }}
                />
              </div>
              <div className="flex-1">
                <div className="text-3xl font-bold text-[#5d9bec]">{ctWinPct}%</div>
                <div className="text-sm text-blue-400 font-medium">CT Side</div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* T Side */}
        <div className="p-5 rounded-xl bg-[#16213e]/90 backdrop-blur-sm border border-yellow-900/30">
          <h2 className="text-lg font-bold text-yellow-400 mb-4 uppercase tracking-wider">
            Terrorists
          </h2>
          <div className="space-y-3">
            <Slider label="Alive" value={state.t_alive} onChange={(v) => set("t_alive", v)} min={0} max={5} side="t" />
            <Slider label="Avg Health" value={state.t_health_avg} onChange={(v) => set("t_health_avg", v)} min={0} max={100} side="t" />
            <Slider label="Armor %" value={state.t_armor_pct} onChange={(v) => set("t_armor_pct", v)} min={0} max={100} side="t" suffix="%" />
            <Slider label="Rifles" value={state.t_rifles} onChange={(v) => set("t_rifles", v)} min={0} max={5} side="t" />
            <Slider label="AWPs" value={state.t_awps} onChange={(v) => set("t_awps", v)} min={0} max={5} side="t" />
            <Slider label="Team Money" value={state.t_money} onChange={(v) => set("t_money", v)} min={0} max={80000} step={500} side="t" suffix="$" />
            <Slider label="Score" value={state.t_score} onChange={(v) => set("t_score", v)} min={0} max={15} side="t" />
          </div>
        </div>

        {/* CT Side */}
        <div className="p-5 rounded-xl bg-[#16213e]/90 backdrop-blur-sm border border-blue-900/30">
          <h2 className="text-lg font-bold text-[#5d9bec] mb-4 uppercase tracking-wider">
            Counter-Terrorists
          </h2>
          <div className="space-y-3">
            <Slider label="Alive" value={state.ct_alive} onChange={(v) => set("ct_alive", v)} min={0} max={5} side="ct" />
            <Slider label="Avg Health" value={state.ct_health_avg} onChange={(v) => set("ct_health_avg", v)} min={0} max={100} side="ct" />
            <Slider label="Armor %" value={state.ct_armor_pct} onChange={(v) => set("ct_armor_pct", v)} min={0} max={100} side="ct" suffix="%" />
            <Slider label="Rifles" value={state.ct_rifles} onChange={(v) => set("ct_rifles", v)} min={0} max={5} side="ct" />
            <Slider label="AWPs" value={state.ct_awps} onChange={(v) => set("ct_awps", v)} min={0} max={5} side="ct" />
            <Slider label="Team Money" value={state.ct_money} onChange={(v) => set("ct_money", v)} min={0} max={80000} step={500} side="ct" suffix="$" />
            <Slider label="Score" value={state.ct_score} onChange={(v) => set("ct_score", v)} min={0} max={15} side="ct" />
          </div>
        </div>
      </div>

      {/* Round State */}
      <div className="p-5 rounded-xl bg-[#16213e]/90 backdrop-blur-sm border border-gray-700/30 mb-8">
        <h2 className="text-lg font-bold text-gray-200 mb-4 uppercase tracking-wider">Round State</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <div>
            <label className="text-sm text-gray-300 block mb-1 font-medium">Map</label>
            <select
              value={state.map}
              onChange={(e) => set("map", e.target.value)}
              className="w-full bg-[#0a0a1a] border border-gray-600 rounded-lg px-3 py-2 text-white"
            >
              {MAPS.map((m) => (
                <option key={m} value={m}>{m.replace("de_", "").charAt(0).toUpperCase() + m.replace("de_", "").slice(1)}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm text-gray-300 block mb-1 font-medium">Round</label>
            <div className="flex items-center gap-2 bg-[#0a0a1a] border border-gray-600 rounded-lg px-3 py-2">
              <input
                type="range"
                min={1}
                max={30}
                value={state.round_num}
                onChange={(e) => set("round_num", Number(e.target.value))}
                className="flex-1 h-2 rounded-lg appearance-none cursor-pointer accent-gray-400"
              />
              <span className="font-mono font-bold text-white w-8 text-right">{state.round_num}</span>
            </div>
          </div>
          <div>
            <label className="text-sm text-gray-300 block mb-1 font-medium">Time Remaining</label>
            <div className="flex items-center gap-2 bg-[#0a0a1a] border border-gray-600 rounded-lg px-3 py-2">
              <input
                type="range"
                min={0}
                max={175}
                value={state.time_remaining}
                onChange={(e) => set("time_remaining", Number(e.target.value))}
                className="flex-1 h-2 rounded-lg appearance-none cursor-pointer accent-gray-400"
              />
              <span className="font-mono font-bold text-white w-12 text-right">
                {Math.floor(state.time_remaining / 60)}:{String(state.time_remaining % 60).padStart(2, "0")}
              </span>
            </div>
          </div>
        </div>
        <div className="mt-4 flex items-center gap-3">
          <label className="text-sm text-gray-300 font-medium">Bomb</label>
          <button
            onClick={() => set("bomb_planted", !state.bomb_planted)}
            className={`px-5 py-2 rounded-lg text-sm font-bold uppercase tracking-wide transition-all ${
              state.bomb_planted
                ? "bg-red-600 text-white shadow-lg shadow-red-600/30"
                : "bg-[#0a0a1a] text-gray-400 border border-gray-600 hover:border-gray-500"
            }`}
          >
            {state.bomb_planted ? "PLANTED" : "Not Planted"}
          </button>
        </div>
      </div>

      {/* Predict Button */}
      <button
        onClick={runPrediction}
        disabled={loading}
        className="w-full py-4 rounded-xl bg-gradient-to-r from-yellow-600 to-yellow-500 hover:from-yellow-500 hover:to-yellow-400 text-black font-bold text-lg uppercase tracking-wider transition-all disabled:opacity-50 cursor-pointer shadow-lg shadow-yellow-600/20"
      >
        {loading ? "Running Model..." : "Predict Round Winner"}
      </button>

      <p className="text-center text-xs text-gray-500 mt-6">
        MLP Neural Network trained on 171K CS2 round snapshots
      </p>
    </main>
  );
}
