import { buildFeatureVector, GameState } from "../app/predict";

// Feature name order from feature_names.json
const FEATURE_NAMES = [
  "ct_alive", "t_alive", "man_advantage",
  "ct_health_total", "t_health_total", "ct_health_avg", "t_health_avg", "health_advantage",
  "ct_armor_total", "t_armor_total", "ct_has_armor_count", "t_has_armor_count",
  "ct_helmet_count", "t_helmet_count", "ct_defuser_count",
  "ct_rifles", "t_rifles", "ct_awps", "t_awps", "ct_snipers", "t_snipers",
  "ct_smgs", "t_smgs", "ct_shotguns", "t_shotguns", "ct_heavy", "t_heavy",
  "ct_has_primary", "t_has_primary", "rifle_advantage", "awp_advantage",
  "ct_pistol_only", "t_pistol_only",
  "ct_flashbangs", "t_flashbangs", "ct_smokes", "t_smokes",
  "ct_hegrenades", "t_hegrenades", "ct_molotovs", "t_molotovs",
  "ct_decoys", "t_decoys", "ct_utility_total", "t_utility_total", "utility_advantage",
  "ct_money_total", "t_money_total", "ct_money_avg", "t_money_avg",
  "ct_equipment_value", "t_equipment_value", "equipment_advantage",
  "ct_equipment_avg", "t_equipment_avg",
  "bomb_planted", "bomb_site", "time_since_plant",
  "time_elapsed", "time_remaining", "round_progress",
  "round_num", "ct_score", "t_score", "score_diff",
  "is_pistol_round", "is_second_round", "ct_round_type", "t_round_type",
  "is_force_buy_ct", "is_force_buy_t",
  "map_de_dust2", "map_de_mirage", "map_de_inferno", "map_de_nuke",
  "map_de_overpass", "map_de_vertigo", "map_de_ancient", "map_de_anubis",
  "ct_avg_rank", "t_avg_rank", "rank_diff", "ct_avg_wins", "t_avg_wins", "wins_diff",
  "kills_this_round_ct", "kills_this_round_t", "first_blood_ct", "first_blood_t",
  "headshot_kills_ct", "headshot_kills_t", "awp_kills_ct", "awp_kills_t",
  "time_since_last_kill",
  "damage_dealt_ct", "damage_dealt_t", "damage_taken_ct", "damage_taken_t",
  "ct_won_last_round", "t_won_last_round",
  "ct_win_streak", "t_win_streak", "ct_loss_streak", "t_loss_streak",
];

function idx(name: string): number {
  const i = FEATURE_NAMES.indexOf(name);
  if (i === -1) throw new Error(`Feature "${name}" not found`);
  return i;
}

function defaultState(): GameState {
  return {
    ct_alive: 5, t_alive: 5,
    ct_health_avg: 100, t_health_avg: 100,
    ct_armor_pct: 100, t_armor_pct: 100,
    ct_rifles: 4, t_rifles: 4,
    ct_awps: 1, t_awps: 1,
    bomb_planted: false, time_remaining: 115,
    round_num: 15, ct_score: 7, t_score: 7,
    map: "de_dust2", ct_money: 20000, t_money: 20000,
  };
}

describe("buildFeatureVector", () => {
  test("returns exactly 104 features", () => {
    const v = buildFeatureVector(defaultState());
    expect(v.length).toBe(104);
    expect(v.length).toBe(FEATURE_NAMES.length);
  });

  test("no NaN or Infinity values", () => {
    const v = buildFeatureVector(defaultState());
    v.forEach((val, i) => {
      expect(Number.isFinite(val)).toBe(true);
    });
  });

  // ==========================================
  // PLAYER COUNTS
  // ==========================================
  describe("player counts and man advantage", () => {
    test("ct_alive and t_alive match input", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 3, t_alive: 4 });
      expect(v[idx("ct_alive")]).toBe(3);
      expect(v[idx("t_alive")]).toBe(4);
    });

    test("man_advantage = t_alive - ct_alive", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 3, t_alive: 5 });
      expect(v[idx("man_advantage")]).toBe(2); // T advantage
    });

    test("man_advantage negative when CT has more", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 5, t_alive: 2 });
      expect(v[idx("man_advantage")]).toBe(-3);
    });
  });

  // ==========================================
  // KILLS (critical: must not be swapped)
  // ==========================================
  describe("kills — CT kills = dead T players", () => {
    test("CT kills 2 Ts: kills_this_round_ct = 2", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 5, t_alive: 3 });
      expect(v[idx("kills_this_round_ct")]).toBe(2); // 5-3 = 2 dead Ts = CT kills
      expect(v[idx("kills_this_round_t")]).toBe(0); // no dead CTs
    });

    test("T kills 3 CTs: kills_this_round_t = 3", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 2, t_alive: 5 });
      expect(v[idx("kills_this_round_ct")]).toBe(0);
      expect(v[idx("kills_this_round_t")]).toBe(3); // 5-2 = 3 dead CTs = T kills
    });

    test("both sides have kills", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 3, t_alive: 4 });
      expect(v[idx("kills_this_round_ct")]).toBe(1); // 1 dead T
      expect(v[idx("kills_this_round_t")]).toBe(2); // 2 dead CTs
    });

    test("no kills when all alive", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 5, t_alive: 5 });
      expect(v[idx("kills_this_round_ct")]).toBe(0);
      expect(v[idx("kills_this_round_t")]).toBe(0);
    });
  });

  // ==========================================
  // FIRST BLOOD
  // ==========================================
  describe("first blood — mutually exclusive", () => {
    test("no kills = no first blood", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 5, t_alive: 5 });
      expect(v[idx("first_blood_ct")]).toBe(0);
      expect(v[idx("first_blood_t")]).toBe(0);
    });

    test("only T dead = CT got first blood", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 5, t_alive: 4 });
      expect(v[idx("first_blood_ct")]).toBe(1);
      expect(v[idx("first_blood_t")]).toBe(0);
    });

    test("only CT dead = T got first blood", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 4, t_alive: 5 });
      expect(v[idx("first_blood_ct")]).toBe(0);
      expect(v[idx("first_blood_t")]).toBe(1);
    });

    test("both sides have kills = only one first blood", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 3, t_alive: 4 });
      const fb_ct = v[idx("first_blood_ct")];
      const fb_t = v[idx("first_blood_t")];
      expect(fb_ct + fb_t).toBe(1); // exactly one
    });
  });

  // ==========================================
  // DAMAGE
  // ==========================================
  describe("damage — ~100 per kill, cross-assigned", () => {
    test("CT kills 2 = damage_dealt_ct ~200", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 5, t_alive: 3 });
      expect(v[idx("damage_dealt_ct")]).toBe(200);
      expect(v[idx("damage_dealt_t")]).toBe(0);
    });

    test("damage_taken_ct = damage_dealt_t", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 3, t_alive: 4 });
      expect(v[idx("damage_taken_ct")]).toBe(v[idx("damage_dealt_t")]);
      expect(v[idx("damage_taken_t")]).toBe(v[idx("damage_dealt_ct")]);
    });
  });

  // ==========================================
  // TIME AND ROUND PROGRESS
  // ==========================================
  describe("time — 115s round duration", () => {
    test("round start: time_elapsed=0, progress=0", () => {
      const v = buildFeatureVector({ ...defaultState(), time_remaining: 115 });
      expect(v[idx("time_elapsed")]).toBe(0);
      expect(v[idx("round_progress")]).toBeCloseTo(0, 2);
    });

    test("half round: ~57.5s elapsed, progress ~0.5", () => {
      const v = buildFeatureVector({ ...defaultState(), time_remaining: 57 });
      expect(v[idx("time_elapsed")]).toBe(58);
      expect(v[idx("round_progress")]).toBeCloseTo(58 / 115, 2);
    });

    test("round end: time_remaining=0, progress=1", () => {
      const v = buildFeatureVector({ ...defaultState(), time_remaining: 0 });
      expect(v[idx("time_elapsed")]).toBe(115);
      expect(v[idx("round_progress")]).toBeCloseTo(1.0, 2);
    });

    test("round_progress clamped to 1.0 max", () => {
      const v = buildFeatureVector({ ...defaultState(), time_remaining: 0 });
      expect(v[idx("round_progress")]).toBeLessThanOrEqual(1.0);
    });
  });

  // ==========================================
  // BOMB
  // ==========================================
  describe("bomb planted", () => {
    test("not planted: bomb features = 0", () => {
      const v = buildFeatureVector({ ...defaultState(), bomb_planted: false });
      expect(v[idx("bomb_planted")]).toBe(0);
      expect(v[idx("time_since_plant")]).toBe(0);
    });

    test("planted: bomb_planted = 1", () => {
      const v = buildFeatureVector({ ...defaultState(), bomb_planted: true });
      expect(v[idx("bomb_planted")]).toBe(1);
      expect(v[idx("time_since_plant")]).toBeGreaterThan(0);
    });
  });

  // ==========================================
  // PISTOL ROUNDS
  // ==========================================
  describe("pistol and second round detection", () => {
    test("round 1 is pistol", () => {
      const v = buildFeatureVector({ ...defaultState(), round_num: 1 });
      expect(v[idx("is_pistol_round")]).toBe(1);
      expect(v[idx("is_second_round")]).toBe(0);
    });

    test("round 13 is second-half pistol", () => {
      const v = buildFeatureVector({ ...defaultState(), round_num: 13 });
      expect(v[idx("is_pistol_round")]).toBe(1);
    });

    test("round 16 is NOT pistol (old CS:GO mistake)", () => {
      const v = buildFeatureVector({ ...defaultState(), round_num: 16 });
      expect(v[idx("is_pistol_round")]).toBe(0);
    });

    test("round 2 is second round", () => {
      const v = buildFeatureVector({ ...defaultState(), round_num: 2 });
      expect(v[idx("is_second_round")]).toBe(1);
    });

    test("round 14 is second round", () => {
      const v = buildFeatureVector({ ...defaultState(), round_num: 14 });
      expect(v[idx("is_second_round")]).toBe(1);
    });

    test("round 15 is neither", () => {
      const v = buildFeatureVector({ ...defaultState(), round_num: 15 });
      expect(v[idx("is_pistol_round")]).toBe(0);
      expect(v[idx("is_second_round")]).toBe(0);
    });
  });

  // ==========================================
  // ROUND TYPE CLASSIFICATION
  // ==========================================
  describe("round type — eco/force/full buy", () => {
    test("eco: avg money < $2000", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_money: 5000 }); // avg = 1000
      expect(v[idx("ct_round_type")]).toBe(0); // eco
      expect(v[idx("is_force_buy_ct")]).toBe(0);
    });

    test("force buy: avg money $2000-3999", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_money: 15000 }); // avg = 3000
      expect(v[idx("ct_round_type")]).toBe(1); // force
      expect(v[idx("is_force_buy_ct")]).toBe(1);
    });

    test("full buy: avg money >= $4000", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_money: 25000 }); // avg = 5000
      expect(v[idx("ct_round_type")]).toBe(2); // full
      expect(v[idx("is_force_buy_ct")]).toBe(0);
    });

    test("$3800 avg is force (not full) — threshold is $4000", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_money: 19000 }); // avg = 3800
      expect(v[idx("ct_round_type")]).toBe(1); // force, not full
    });
  });

  // ==========================================
  // MAP ONE-HOT
  // ==========================================
  describe("map one-hot encoding", () => {
    test("dust2 selected: only dust2 = 1", () => {
      const v = buildFeatureVector({ ...defaultState(), map: "de_dust2" });
      expect(v[idx("map_de_dust2")]).toBe(1);
      expect(v[idx("map_de_mirage")]).toBe(0);
      expect(v[idx("map_de_inferno")]).toBe(0);
    });

    test("inferno selected: only inferno = 1", () => {
      const v = buildFeatureVector({ ...defaultState(), map: "de_inferno" });
      expect(v[idx("map_de_dust2")]).toBe(0);
      expect(v[idx("map_de_inferno")]).toBe(1);
    });

    test("exactly one map is 1", () => {
      const v = buildFeatureVector(defaultState());
      const mapSum = [71, 72, 73, 74, 75, 76, 77, 78].reduce((s, i) => s + v[i], 0);
      expect(mapSum).toBe(1);
    });
  });

  // ==========================================
  // EQUIPMENT VALUE
  // ==========================================
  describe("equipment value — estimated from loadout", () => {
    test("full buy has high equipment value", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_rifles: 4, ct_awps: 1, ct_armor_pct: 100 });
      const equip = v[idx("ct_equipment_value")];
      expect(equip).toBeGreaterThan(10000); // rifles + AWP + armor
    });

    test("eco has low equipment value", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_rifles: 0, ct_awps: 0, ct_armor_pct: 0 });
      const equip = v[idx("ct_equipment_value")];
      expect(equip).toBeLessThan(5000); // pistols only
    });

    test("equipment_advantage = t_equip - ct_equip", () => {
      const v = buildFeatureVector(defaultState());
      const diff = v[idx("t_equipment_value")] - v[idx("ct_equipment_value")];
      expect(v[idx("equipment_advantage")]).toBeCloseTo(diff, 1);
    });
  });

  // ==========================================
  // UTILITY
  // ==========================================
  describe("utility — scales with buy type", () => {
    test("eco round: no utility", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_money: 5000 }); // avg 1000 = eco
      expect(v[idx("ct_flashbangs")]).toBe(0);
      expect(v[idx("ct_smokes")]).toBe(0);
      expect(v[idx("ct_utility_total")]).toBe(0);
    });

    test("full buy: has utility", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_money: 25000, ct_alive: 5 });
      expect(v[idx("ct_flashbangs")]).toBeGreaterThan(0);
      expect(v[idx("ct_smokes")]).toBeGreaterThan(0);
      expect(v[idx("ct_utility_total")]).toBeGreaterThan(0);
    });

    test("utility_advantage = t_total - ct_total", () => {
      const v = buildFeatureVector(defaultState());
      const diff = v[idx("t_utility_total")] - v[idx("ct_utility_total")];
      expect(v[idx("utility_advantage")]).toBe(diff);
    });
  });

  // ==========================================
  // SCORE
  // ==========================================
  describe("score features", () => {
    test("score_diff = ct_score - t_score", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_score: 10, t_score: 5 });
      expect(v[idx("score_diff")]).toBe(5);
    });

    test("negative score_diff when T leads", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_score: 3, t_score: 8 });
      expect(v[idx("score_diff")]).toBe(-5);
    });
  });

  // ==========================================
  // HEALTH
  // ==========================================
  describe("health features", () => {
    test("health_advantage = t_total - ct_total", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 5, t_alive: 5, ct_health_avg: 80, t_health_avg: 100 });
      expect(v[idx("health_advantage")]).toBe(5 * 100 - 5 * 80); // 100
    });

    test("health total = alive * avg", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 3, ct_health_avg: 50 });
      expect(v[idx("ct_health_total")]).toBe(150);
    });
  });

  // ==========================================
  // WEAPONS
  // ==========================================
  describe("weapon features", () => {
    test("rifle_advantage = t_rifles - ct_rifles", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_rifles: 3, t_rifles: 5 });
      expect(v[idx("rifle_advantage")]).toBe(2);
    });

    test("awp_advantage = t_awps - ct_awps", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_awps: 2, t_awps: 0 });
      expect(v[idx("awp_advantage")]).toBe(-2);
    });

    test("pistol_only = alive - primary", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 5, ct_rifles: 2, ct_awps: 1 });
      expect(v[idx("ct_has_primary")]).toBe(3);
      expect(v[idx("ct_pistol_only")]).toBe(2);
    });
  });

  // ==========================================
  // DEFUSERS
  // ==========================================
  describe("defuser count scales with economy", () => {
    test("eco: 0 defusers", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_money: 5000 }); // avg 1000
      expect(v[idx("ct_defuser_count")]).toBe(0);
    });

    test("full buy: multiple defusers", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_money: 25000, ct_alive: 5 }); // avg 5000
      expect(v[idx("ct_defuser_count")]).toBeGreaterThan(0);
    });
  });

  // ==========================================
  // SYMMETRY TEST — equal state should be ~symmetric
  // ==========================================
  describe("symmetry", () => {
    test("equal teams: advantages should be 0", () => {
      const v = buildFeatureVector(defaultState());
      expect(v[idx("man_advantage")]).toBe(0);
      expect(v[idx("health_advantage")]).toBe(0);
      expect(v[idx("rifle_advantage")]).toBe(0);
      expect(v[idx("awp_advantage")]).toBe(0);
      expect(v[idx("score_diff")]).toBe(0);
    });
  });

  // ==========================================
  // EDGE CASES
  // ==========================================
  describe("edge cases", () => {
    test("all players dead on one side", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 0, t_alive: 5 });
      expect(v[idx("ct_alive")]).toBe(0);
      expect(v[idx("kills_this_round_t")]).toBe(5);
      expect(v[idx("ct_health_total")]).toBe(0);
    });

    test("all players dead both sides", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_alive: 0, t_alive: 0 });
      expect(v[idx("man_advantage")]).toBe(0);
      expect(v[idx("kills_this_round_ct")]).toBe(5);
      expect(v[idx("kills_this_round_t")]).toBe(5);
    });

    test("zero money", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_money: 0, t_money: 0 });
      expect(v[idx("ct_money_total")]).toBe(0);
      expect(v[idx("ct_round_type")]).toBe(0); // eco
    });

    test("max money", () => {
      const v = buildFeatureVector({ ...defaultState(), ct_money: 80000 });
      expect(v[idx("ct_money_total")]).toBe(80000);
      expect(v[idx("ct_money_avg")]).toBe(16000);
    });
  });
});
