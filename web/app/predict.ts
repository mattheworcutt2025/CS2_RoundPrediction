import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;
let scalerMean: number[] | null = null;
let scalerStd: number[] | null = null;

async function loadModel() {
  if (!session) {
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create("/mlp_tuned.onnx");
    const scaler = await fetch("/scaler_params.json").then((r) => r.json());
    scalerMean = scaler.mean;
    scalerStd = scaler.std;
  }
}

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

function buildFeatureVector(state: GameState): number[] {
  const ct_alive = state.ct_alive;
  const t_alive = state.t_alive;
  const ct_health_total = ct_alive * state.ct_health_avg;
  const t_health_total = t_alive * state.t_health_avg;
  const ct_armor_total = ct_alive * state.ct_armor_pct;
  const t_armor_total = t_alive * state.t_armor_pct;
  const ct_equip = state.ct_money * 0.6;
  const t_equip = state.t_money * 0.6;
  const time_elapsed = 175 - state.time_remaining;
  const round_progress = time_elapsed / 175;
  const is_pistol = state.round_num === 1 || state.round_num === 16 ? 1 : 0;
  const is_second = state.round_num === 2 || state.round_num === 17 ? 1 : 0;

  const ct_has_primary = Math.min(ct_alive, state.ct_rifles + state.ct_awps);
  const t_has_primary = Math.min(t_alive, state.t_rifles + state.t_awps);
  const ct_pistol_only = ct_alive - ct_has_primary;
  const t_pistol_only = t_alive - t_has_primary;

  function classifyRound(money: number): number {
    const avg = money / 5;
    if (avg < 2000) return 0; // eco
    if (avg < 3500) return 1; // force
    return 2; // full buy
  }

  const ct_round_type = classifyRound(state.ct_money);
  const t_round_type = classifyRound(state.t_money);

  // Map one-hot
  const maps = ["de_dust2", "de_mirage", "de_inferno", "de_nuke", "de_overpass", "de_vertigo", "de_ancient", "de_anubis"];
  const mapOneHot = maps.map((m) => (m === state.map ? 1 : 0));

  // 104 features in order matching feature_names.json
  return [
    ct_alive,                                    // ct_alive
    t_alive,                                     // t_alive
    t_alive - ct_alive,                          // man_advantage
    ct_health_total,                             // ct_health_total
    t_health_total,                              // t_health_total
    state.ct_health_avg,                         // ct_health_avg
    state.t_health_avg,                          // t_health_avg
    t_health_total - ct_health_total,            // health_advantage
    ct_armor_total,                              // ct_armor_total
    t_armor_total,                               // t_armor_total
    Math.min(ct_alive, Math.round(ct_alive * state.ct_armor_pct / 100)), // ct_has_armor_count
    Math.min(t_alive, Math.round(t_alive * state.t_armor_pct / 100)),   // t_has_armor_count
    Math.min(ct_alive, Math.round(ct_alive * state.ct_armor_pct / 100)), // ct_helmet_count
    Math.min(t_alive, Math.round(t_alive * state.t_armor_pct / 100)),   // t_helmet_count
    Math.min(ct_alive, 2),                       // ct_defuser_count
    state.ct_rifles,                             // ct_rifles
    state.t_rifles,                              // t_rifles
    state.ct_awps,                               // ct_awps
    state.t_awps,                                // t_awps
    state.ct_awps,                               // ct_snipers
    state.t_awps,                                // t_snipers
    0,                                           // ct_smgs
    0,                                           // t_smgs
    0,                                           // ct_shotguns
    0,                                           // t_shotguns
    0,                                           // ct_heavy
    0,                                           // t_heavy
    ct_has_primary,                              // ct_has_primary
    t_has_primary,                               // t_has_primary
    state.t_rifles - state.ct_rifles,            // rifle_advantage
    state.t_awps - state.ct_awps,                // awp_advantage
    ct_pistol_only,                              // ct_pistol_only
    t_pistol_only,                               // t_pistol_only
    Math.min(ct_alive, 2),                       // ct_flashbangs
    Math.min(t_alive, 2),                        // t_flashbangs
    Math.min(ct_alive, 1),                       // ct_smokes
    Math.min(t_alive, 1),                        // t_smokes
    Math.min(ct_alive, 1),                       // ct_hegrenades
    Math.min(t_alive, 1),                        // t_hegrenades
    Math.min(ct_alive, 1),                       // ct_molotovs
    Math.min(t_alive, 1),                        // t_molotovs
    0,                                           // ct_decoys
    0,                                           // t_decoys
    ct_alive * 4,                                // ct_utility_total
    t_alive * 4,                                 // t_utility_total
    0,                                           // utility_advantage
    state.ct_money,                              // ct_money_total
    state.t_money,                               // t_money_total
    state.ct_money / 5,                          // ct_money_avg
    state.t_money / 5,                           // t_money_avg
    ct_equip,                                    // ct_equipment_value
    t_equip,                                     // t_equipment_value
    t_equip - ct_equip,                          // equipment_advantage
    ct_equip / 5,                                // ct_equipment_avg
    t_equip / 5,                                 // t_equipment_avg
    state.bomb_planted ? 1 : 0,                  // bomb_planted
    state.bomb_planted ? 1 : 0,                  // bomb_site (A=0, B=1)
    state.bomb_planted ? 10 : 0,                 // time_since_plant
    time_elapsed,                                // time_elapsed
    state.time_remaining,                        // time_remaining
    round_progress,                              // round_progress
    state.round_num,                             // round_num
    state.ct_score,                              // ct_score
    state.t_score,                               // t_score
    state.ct_score - state.t_score,              // score_diff
    is_pistol,                                   // is_pistol_round
    is_second,                                   // is_second_round
    ct_round_type,                               // ct_round_type
    t_round_type,                                // t_round_type
    ct_round_type === 1 ? 1 : 0,                // is_force_buy_ct
    t_round_type === 1 ? 1 : 0,                 // is_force_buy_t
    ...mapOneHot,                                // 8 map features
    10,                                          // ct_avg_rank
    10,                                          // t_avg_rank
    0,                                           // rank_diff
    50,                                          // ct_avg_wins
    50,                                          // t_avg_wins
    0,                                           // wins_diff
    5 - ct_alive,                                // kills_this_round_ct (T killed CTs)
    5 - t_alive,                                 // kills_this_round_t (CT killed Ts)
    5 - ct_alive > 0 ? 1 : 0,                   // first_blood_ct
    5 - t_alive > 0 ? 1 : 0,                    // first_blood_t
    Math.floor((5 - ct_alive) * 0.4),            // headshot_kills_ct
    Math.floor((5 - t_alive) * 0.4),             // headshot_kills_t
    0,                                           // awp_kills_ct
    0,                                           // awp_kills_t
    time_elapsed > 0 ? 5 : 0,                   // time_since_last_kill
    (5 - t_alive) * 30,                          // damage_dealt_ct
    (5 - ct_alive) * 30,                         // damage_dealt_t
    (5 - ct_alive) * 30,                         // damage_taken_ct
    (5 - t_alive) * 30,                          // damage_taken_t
    0,                                           // ct_won_last_round
    0,                                           // t_won_last_round
    0,                                           // ct_win_streak
    0,                                           // t_win_streak
    0,                                           // ct_loss_streak
    0,                                           // t_loss_streak
  ];
}

export async function predict(state: GameState): Promise<number> {
  await loadModel();

  const raw = buildFeatureVector(state);

  // StandardScaler: (x - mean) / std
  const scaled = raw.map((val, i) => {
    const std = scalerStd![i];
    if (std === 0) return 0;
    return (val - scalerMean![i]) / std;
  });

  const tensor = new ort.Tensor("float32", Float32Array.from(scaled), [1, 104]);
  const results = await session!.run({ features: tensor });
  const prob = (results.probability.data as Float32Array)[0];
  return prob; // probability of CT win
}
