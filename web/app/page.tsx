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
  color,
  suffix = "",
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  color: string;
  suffix?: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="w-32 text-sm text-gray-400 shrink-0">{label}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`flex-1 h-2 rounded-lg appearance-none cursor-pointer accent-${color}-500`}
      />
      <span className={`w-16 text-right font-mono text-${color}-400`}>
        {value}{suffix}
      </span>
    </div>
  );
}

export default function Home() {
  const [state, setState] = useState<GameState>(defaultState);
  const [result, setResult] = useState<{ ct: number; t: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelReady, setModelReady] = useState(false);

  const set = useCallback(
    <K extends keyof GameState>(key: K, val: GameState[K]) => {
      setState((prev) => ({ ...prev, [key]: val }));
      setResult(null);
    },
    []
  );

  const runPrediction = async () => {
    setLoading(true);
    try {
      const prob = await predict(state);
      setResult({ ct: prob, t: 1 - prob });
      if (!modelReady) setModelReady(true);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const ctWinPct = result ? Math.round(result.ct * 100) : null;
  const tWinPct = result ? Math.round(result.t * 100) : null;

  return (
    <main className="max-w-4xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-2">
          CS2 Round Predictor
        </h1>
        <p className="text-gray-400">
          Deep learning model with 96.71% accuracy
          <span className="text-gray-600"> | </span>
          <span className="text-gray-500">104 features, 1.2M parameters, runs in your browser</span>
        </p>
      </div>

      {result && (
        <div className="mb-8 p-6 rounded-2xl bg-gray-900 border border-gray-800">
          <div className="flex items-center gap-4 mb-4">
            <div className="flex-1 text-right">
              <div className="text-3xl font-bold text-blue-400">{tWinPct}%</div>
              <div className="text-sm text-gray-400">T Side</div>
            </div>
            <div className="w-full max-w-md h-8 bg-gray-800 rounded-full overflow-hidden flex">
              <div
                className="h-full bg-gradient-to-r from-orange-600 to-orange-500 transition-all duration-500"
                style={{ width: `${tWinPct}%` }}
              />
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-500"
                style={{ width: `${ctWinPct}%` }}
              />
            </div>
            <div className="flex-1">
              <div className="text-3xl font-bold text-cyan-400">{ctWinPct}%</div>
              <div className="text-sm text-gray-400">CT Side</div>
            </div>
          </div>
          <div className="text-center text-sm text-gray-500">
            {ctWinPct! > 60
              ? "CT side has a strong advantage"
              : ctWinPct! > 50
              ? "Slight CT advantage"
              : tWinPct! > 60
              ? "T side has a strong advantage"
              : tWinPct! > 50
              ? "Slight T advantage"
              : "This round is a coin flip"}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* T Side */}
        <div className="p-5 rounded-xl bg-gray-900 border border-orange-900/30">
          <h2 className="text-lg font-semibold text-orange-400 mb-4">Terrorists</h2>
          <div className="space-y-3">
            <Slider label="Alive" value={state.t_alive} onChange={(v) => set("t_alive", v)} min={0} max={5} color="orange" />
            <Slider label="Avg Health" value={state.t_health_avg} onChange={(v) => set("t_health_avg", v)} min={0} max={100} color="orange" />
            <Slider label="Armor %" value={state.t_armor_pct} onChange={(v) => set("t_armor_pct", v)} min={0} max={100} color="orange" suffix="%" />
            <Slider label="Rifles" value={state.t_rifles} onChange={(v) => set("t_rifles", v)} min={0} max={5} color="orange" />
            <Slider label="AWPs" value={state.t_awps} onChange={(v) => set("t_awps", v)} min={0} max={5} color="orange" />
            <Slider label="Team Money" value={state.t_money} onChange={(v) => set("t_money", v)} min={0} max={80000} step={500} color="orange" suffix="$" />
            <Slider label="Score" value={state.t_score} onChange={(v) => set("t_score", v)} min={0} max={15} color="orange" />
          </div>
        </div>

        {/* CT Side */}
        <div className="p-5 rounded-xl bg-gray-900 border border-cyan-900/30">
          <h2 className="text-lg font-semibold text-cyan-400 mb-4">Counter-Terrorists</h2>
          <div className="space-y-3">
            <Slider label="Alive" value={state.ct_alive} onChange={(v) => set("ct_alive", v)} min={0} max={5} color="cyan" />
            <Slider label="Avg Health" value={state.ct_health_avg} onChange={(v) => set("ct_health_avg", v)} min={0} max={100} color="cyan" />
            <Slider label="Armor %" value={state.ct_armor_pct} onChange={(v) => set("ct_armor_pct", v)} min={0} max={100} color="cyan" suffix="%" />
            <Slider label="Rifles" value={state.ct_rifles} onChange={(v) => set("ct_rifles", v)} min={0} max={5} color="cyan" />
            <Slider label="AWPs" value={state.ct_awps} onChange={(v) => set("ct_awps", v)} min={0} max={5} color="cyan" />
            <Slider label="Team Money" value={state.ct_money} onChange={(v) => set("ct_money", v)} min={0} max={80000} step={500} color="cyan" suffix="$" />
            <Slider label="Score" value={state.ct_score} onChange={(v) => set("ct_score", v)} min={0} max={15} color="cyan" />
          </div>
        </div>
      </div>

      {/* Game State */}
      <div className="p-5 rounded-xl bg-gray-900 border border-gray-800 mb-8">
        <h2 className="text-lg font-semibold text-gray-300 mb-4">Round State</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="text-sm text-gray-400 block mb-1">Map</label>
            <select
              value={state.map}
              onChange={(e) => set("map", e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white"
            >
              {MAPS.map((m) => (
                <option key={m} value={m}>{m.replace("de_", "").charAt(0).toUpperCase() + m.replace("de_", "").slice(1)}</option>
              ))}
            </select>
          </div>
          <Slider label="Round #" value={state.round_num} onChange={(v) => set("round_num", v)} min={1} max={30} color="gray" />
          <Slider label="Time Left" value={state.time_remaining} onChange={(v) => set("time_remaining", v)} min={0} max={175} color="gray" suffix="s" />
        </div>
        <div className="mt-4 flex items-center gap-3">
          <label className="text-sm text-gray-400">Bomb Planted</label>
          <button
            onClick={() => set("bomb_planted", !state.bomb_planted)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              state.bomb_planted
                ? "bg-red-600 text-white"
                : "bg-gray-800 text-gray-400 border border-gray-700"
            }`}
          >
            {state.bomb_planted ? "PLANTED" : "Not Planted"}
          </button>
        </div>
      </div>

      <button
        onClick={runPrediction}
        disabled={loading}
        className="w-full py-4 rounded-xl bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 font-bold text-lg transition-all disabled:opacity-50 cursor-pointer"
      >
        {loading ? "Running Model..." : "Predict Round Winner"}
      </button>

      <p className="text-center text-xs text-gray-600 mt-6">
        MLP Neural Network (104 features, [896-704-448-448]) trained on 171K CS2 round snapshots
        <br />
        Model runs entirely in your browser via ONNX Runtime
      </p>
    </main>
  );
}
