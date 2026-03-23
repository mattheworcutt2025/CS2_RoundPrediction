import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;
let scalerMean: number[] | null = null;
let scalerStd: number[] | null = null;

// Python-compatible rounding (banker's rounding / round-half-to-even)
// JS Math.round(2.5) = 3, but Python round(2.5) = 2
// The model was trained with Python's rounding, so we must match it.
function pyRound(n: number): number {
  const floor = Math.floor(n);
  const decimal = n - floor;
  if (Math.abs(decimal - 0.5) < 1e-9) {
    // Exactly .5 — round to even
    return floor % 2 === 0 ? floor : floor + 1;
  }
  return Math.round(n);
}

async function loadModel() {
  if (!session) {
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create("/mlp_tuned.onnx");
    const scaler = await fetch("/scaler_params.json").then((r) => r.json());
    scalerMean = scaler.mean;
    scalerStd = scaler.std;
  }
}

export interface GameState {
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

// Estimate equipment value from weapons + armor
// Rifle ~$2700-3100, AWP ~$4750, armor+helmet ~$1000, pistol ~$500
function estimateEquipmentValue(
  alive: number, rifles: number, awps: number, armorPct: number
): number {
  const rifleValue = rifles * 2900;
  const awpValue = awps * 4750;
  const armoredPlayers = pyRound(alive * armorPct / 100);
  const armorValue = armoredPlayers * 1000;
  const pistolPlayers = alive - rifles - awps;
  const pistolValue = Math.max(0, pistolPlayers) * 500;
  return rifleValue + awpValue + armorValue + pistolValue;
}

function classifyRound(avgMoney: number): number {
  if (avgMoney < 2000) return 0; // eco
  if (avgMoney < 4000) return 1; // force
  return 2; // full buy
}

export function buildFeatureVector(state: GameState): number[] {
  const ct_alive = state.ct_alive;
  const t_alive = state.t_alive;

  // Health
  const ct_health_total = ct_alive * state.ct_health_avg;
  const t_health_total = t_alive * state.t_health_avg;

  // Armor (100 armor per player max)
  const ct_armor_total = ct_alive * state.ct_armor_pct; // pct acts as avg armor value
  const t_armor_total = t_alive * state.t_armor_pct;
  const ct_armored = pyRound(ct_alive * Math.min(state.ct_armor_pct, 100) / 100);
  const t_armored = pyRound(t_alive * Math.min(state.t_armor_pct, 100) / 100);
  // Helmets: in full buys most armored players have helmets, eco rounds fewer
  const ct_helmets = state.ct_armor_pct >= 50 ? ct_armored : Math.floor(ct_armored * 0.5);
  const t_helmets = state.t_armor_pct >= 50 ? t_armored : Math.floor(t_armored * 0.5);
  // Defusers: typically 1-2 in buys, 0 in ecos
  const ct_money_avg = state.ct_money / 5;
  const ct_defusers = ct_money_avg >= 4000 ? Math.min(ct_alive, 3) : ct_money_avg >= 2000 ? Math.min(ct_alive, 1) : 0;

  // Weapons
  const ct_has_primary = Math.min(ct_alive, state.ct_rifles + state.ct_awps);
  const t_has_primary = Math.min(t_alive, state.t_rifles + state.t_awps);
  const ct_pistol_only = ct_alive - ct_has_primary;
  const t_pistol_only = t_alive - t_has_primary;

  // Equipment value (estimated from loadout)
  const ct_equip = estimateEquipmentValue(ct_alive, state.ct_rifles, state.ct_awps, state.ct_armor_pct);
  const t_equip = estimateEquipmentValue(t_alive, state.t_rifles, state.t_awps, state.t_armor_pct);

  // Utility: estimate based on buy type
  const ct_util_per = ct_money_avg >= 4000 ? 4 : ct_money_avg >= 2000 ? 2 : 0;
  const t_money_avg = state.t_money / 5;
  const t_util_per = t_money_avg >= 4000 ? 4 : t_money_avg >= 2000 ? 2 : 0;
  const ct_flash = Math.min(ct_alive * 2, ct_alive * ct_util_per > 0 ? ct_alive : 0);
  const t_flash = Math.min(t_alive * 2, t_alive * t_util_per > 0 ? t_alive : 0);
  const ct_smoke = ct_util_per >= 4 ? ct_alive : ct_util_per >= 2 ? Math.ceil(ct_alive / 2) : 0;
  const t_smoke = t_util_per >= 4 ? t_alive : t_util_per >= 2 ? Math.ceil(t_alive / 2) : 0;
  const ct_he = ct_util_per >= 4 ? Math.ceil(ct_alive / 2) : 0;
  const t_he = t_util_per >= 4 ? Math.ceil(t_alive / 2) : 0;
  const ct_molly = ct_util_per >= 4 ? Math.ceil(ct_alive / 2) : 0;
  const t_molly = t_util_per >= 4 ? Math.ceil(t_alive / 2) : 0;
  const ct_utility_total = ct_flash + ct_smoke + ct_he + ct_molly;
  const t_utility_total = t_flash + t_smoke + t_he + t_molly;

  // Time (CS2 round = 115 seconds)
  const ROUND_DURATION = 115;
  const BOMB_TIMER = 40;
  const time_remaining = state.bomb_planted
    ? Math.max(0, BOMB_TIMER - 10) // default 10s since plant
    : state.time_remaining;
  const time_elapsed = state.bomb_planted
    ? ROUND_DURATION - state.time_remaining + 10
    : ROUND_DURATION - state.time_remaining;
  const round_progress = Math.min(1.0, Math.max(0, time_elapsed) / ROUND_DURATION);

  // Round type
  const is_pistol = state.round_num === 1 || state.round_num === 13 ? 1 : 0;
  const is_second = state.round_num === 2 || state.round_num === 14 ? 1 : 0;
  const ct_round_type = classifyRound(ct_money_avg);
  const t_round_type = classifyRound(t_money_avg);

  // Map one-hot
  const maps = ["de_dust2", "de_mirage", "de_inferno", "de_nuke", "de_overpass", "de_vertigo", "de_ancient", "de_anubis"];
  const mapOneHot = maps.map((m) => (m === state.map ? 1 : 0));

  // Kills: CT kills = dead T players, T kills = dead CT players
  const ct_kills = 5 - t_alive; // CT killed T players
  const t_kills = 5 - ct_alive; // T killed CT players

  // First blood: only one team gets it
  let first_blood_ct = 0;
  let first_blood_t = 0;
  if (ct_kills > 0 || t_kills > 0) {
    // If more Ts dead, CT likely got first blood; otherwise T did
    if (ct_kills >= t_kills) first_blood_ct = 1;
    else first_blood_t = 1;
  }

  // Damage: ~100 HP per kill on average
  const damage_ct = ct_kills * 100;
  const damage_t = t_kills * 100;

  // Time since last kill
  const any_kills = ct_kills + t_kills;
  const time_since_last_kill = any_kills > 0 ? Math.max(1, time_elapsed * 0.2) : time_elapsed;

  // 104 features in order matching feature_names.json
  return [
    ct_alive,                                    // 0: ct_alive
    t_alive,                                     // 1: t_alive
    t_alive - ct_alive,                          // 2: man_advantage
    ct_health_total,                             // 3: ct_health_total
    t_health_total,                              // 4: t_health_total
    state.ct_health_avg,                         // 5: ct_health_avg
    state.t_health_avg,                          // 6: t_health_avg
    t_health_total - ct_health_total,            // 7: health_advantage
    ct_armor_total,                              // 8: ct_armor_total
    t_armor_total,                               // 9: t_armor_total
    ct_armored,                                  // 10: ct_has_armor_count
    t_armored,                                   // 11: t_has_armor_count
    ct_helmets,                                  // 12: ct_helmet_count
    t_helmets,                                   // 13: t_helmet_count
    ct_defusers,                                 // 14: ct_defuser_count
    state.ct_rifles,                             // 15: ct_rifles
    state.t_rifles,                              // 16: t_rifles
    state.ct_awps,                               // 17: ct_awps
    state.t_awps,                                // 18: t_awps
    state.ct_awps,                               // 19: ct_snipers (AWP + scout, approx)
    state.t_awps,                                // 20: t_snipers
    0,                                           // 21: ct_smgs
    0,                                           // 22: t_smgs
    0,                                           // 23: ct_shotguns
    0,                                           // 24: t_shotguns
    0,                                           // 25: ct_heavy
    0,                                           // 26: t_heavy
    ct_has_primary,                              // 27: ct_has_primary
    t_has_primary,                               // 28: t_has_primary
    state.t_rifles - state.ct_rifles,            // 29: rifle_advantage
    state.t_awps - state.ct_awps,                // 30: awp_advantage
    ct_pistol_only,                              // 31: ct_pistol_only
    t_pistol_only,                               // 32: t_pistol_only
    ct_flash,                                    // 33: ct_flashbangs
    t_flash,                                     // 34: t_flashbangs
    ct_smoke,                                    // 35: ct_smokes
    t_smoke,                                     // 36: t_smokes
    ct_he,                                       // 37: ct_hegrenades
    t_he,                                        // 38: t_hegrenades
    ct_molly,                                    // 39: ct_molotovs
    t_molly,                                     // 40: t_molotovs
    0,                                           // 41: ct_decoys
    0,                                           // 42: t_decoys
    ct_utility_total,                            // 43: ct_utility_total
    t_utility_total,                             // 44: t_utility_total
    t_utility_total - ct_utility_total,          // 45: utility_advantage
    state.ct_money,                              // 46: ct_money_total
    state.t_money,                               // 47: t_money_total
    ct_money_avg,                                // 48: ct_money_avg
    t_money_avg,                                 // 49: t_money_avg
    ct_equip,                                    // 50: ct_equipment_value
    t_equip,                                     // 51: t_equipment_value
    t_equip - ct_equip,                          // 52: equipment_advantage
    ct_equip / 5,                                // 53: ct_equipment_avg
    t_equip / 5,                                 // 54: t_equipment_avg
    state.bomb_planted ? 1 : 0,                  // 55: bomb_planted
    state.bomb_planted ? 0 : 0,                  // 56: bomb_site (default A=0)
    state.bomb_planted ? 10 : 0,                 // 57: time_since_plant
    Math.max(0, time_elapsed),                   // 58: time_elapsed
    Math.max(0, time_remaining),                 // 59: time_remaining
    round_progress,                              // 60: round_progress
    state.round_num,                             // 61: round_num
    state.ct_score,                              // 62: ct_score
    state.t_score,                               // 63: t_score
    state.ct_score - state.t_score,              // 64: score_diff
    is_pistol,                                   // 65: is_pistol_round
    is_second,                                   // 66: is_second_round
    ct_round_type,                               // 67: ct_round_type
    t_round_type,                                // 68: t_round_type
    ct_round_type === 1 ? 1 : 0,                // 69: is_force_buy_ct
    t_round_type === 1 ? 1 : 0,                 // 70: is_force_buy_t
    ...mapOneHot,                                // 71-78: map features
    10,                                          // 79: ct_avg_rank (neutral default)
    10,                                          // 80: t_avg_rank
    0,                                           // 81: rank_diff
    50,                                          // 82: ct_avg_wins
    50,                                          // 83: t_avg_wins
    0,                                           // 84: wins_diff
    ct_kills,                                    // 85: kills_this_round_ct
    t_kills,                                     // 86: kills_this_round_t
    first_blood_ct,                              // 87: first_blood_ct
    first_blood_t,                               // 88: first_blood_t
    pyRound(ct_kills * 0.4),                      // 89: headshot_kills_ct
    pyRound(t_kills * 0.4),                      // 90: headshot_kills_t
    0,                                           // 91: awp_kills_ct
    0,                                           // 92: awp_kills_t
    time_since_last_kill,                        // 93: time_since_last_kill
    damage_ct,                                   // 94: damage_dealt_ct
    damage_t,                                    // 95: damage_dealt_t
    damage_t,                                    // 96: damage_taken_ct (= damage T dealt)
    damage_ct,                                   // 97: damage_taken_t (= damage CT dealt)
    0,                                           // 98: ct_won_last_round
    0,                                           // 99: t_won_last_round
    0,                                           // 100: ct_win_streak
    0,                                           // 101: t_win_streak
    0,                                           // 102: ct_loss_streak
    0,                                           // 103: t_loss_streak
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
