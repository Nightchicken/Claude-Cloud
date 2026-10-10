//! Evolving weather (moisture, cloud type, wind) and lightning driven by cloud water content.
//! Strikes are placed at the wettest cloud columns (via the CPU mirror of the cloud field)
//! and rendered by atmosphere.fs.

const std = @import("std");
const rl = @import("raylib");
const cloudfield = @import("cloudfield.zig");

/// Meters per world unit; matches WORLD_SCALE in atmosphere.fs.
pub const world_scale: f32 = 100.0;
/// Must match BOLT_MAX_SEGMENTS in atmosphere.fs.
pub const max_bolt_segments = 40;

const main_channel_points = 24;
const max_branch_segments = 8;
const strike_search_samples = 16;
const strike_search_radius: f32 = 130; // world units around the origin

/// Slowly drifting weather: every 30-90 s new targets are picked and the state eases toward them.
pub const Weather = struct {
    rng: std.Random.DefaultPrng,
    water: f32 = 0.4,
    water_target: f32 = 0.4,
    /// 0 = stratus, 0.5 = cumulus, 1 = cumulonimbus.
    cloud_type: f32 = 0.5,
    type_target: f32 = 0.5,
    wind_speed: f32 = 15,
    speed_target: f32 = 15,
    wind_angle: f32 = 0.38,
    angle_target: f32 = 0.38,
    /// Accumulated cloud drift in meters; integrating the wind keeps clouds continuous when it changes.
    offset: [2]f32 = .{ 0, 0 },
    next_change: f32 = 30,

    /// Seconds for the state to close ~63% of the gap to its targets.
    const ease_time: f32 = 20;

    pub fn init(seed: u64) Weather {
        return .{ .rng = std.Random.DefaultPrng.init(seed) };
    }

    pub fn update(self: *Weather, dt: f32) void {
        self.next_change -= dt;
        if (self.next_change <= 0) self.pickTargets();

        const k = 1 - @exp(-dt / ease_time);
        self.water += (self.water_target - self.water) * k;
        self.cloud_type += (self.type_target - self.cloud_type) * k;
        self.wind_speed += (self.speed_target - self.wind_speed) * k;
        self.wind_angle += (self.angle_target - self.wind_angle) * k;

        const w = self.wind();
        self.offset[0] += w[0] * dt;
        self.offset[1] += w[1] * dt;
    }

    fn pickTargets(self: *Weather) void {
        const r = self.rng.random();
        self.next_change = 30 + 60 * r.float(f32);
        self.water_target = r.float(f32);
        // Wet air tends to build towering clouds; dry air stays flat or fair-weather puffy.
        self.type_target = std.math.clamp(0.1 + 0.8 * self.water_target + (r.float(f32) - 0.5) * 0.6, 0, 1);
        self.speed_target = 4 + 26 * r.float(f32);
        self.angle_target = self.wind_angle + (r.float(f32) - 0.5) * 1.5;
    }

    /// Manual moisture control; shifts the target too so the change sticks.
    pub fn nudgeWater(self: *Weather, delta: f32) void {
        self.water = std.math.clamp(self.water + delta, 0, 1);
        self.water_target = std.math.clamp(self.water_target + delta, 0, 1);
    }

    /// Wind velocity (x, z) in m/s.
    pub fn wind(self: Weather) [2]f32 {
        return .{ @cos(self.wind_angle) * self.wind_speed, @sin(self.wind_angle) * self.wind_speed };
    }

    pub fn params(self: Weather) cloudfield.Params {
        return cloudfield.Params.init(self.water, self.cloud_type, self.offset, .{ @cos(self.wind_angle), @sin(self.wind_angle) });
    }

    pub fn typeName(self: Weather) [:0]const u8 {
        if (self.cloud_type < 0.25) return "stratus";
        if (self.cloud_type < 0.7) return "cumulus";
        return "cumulonimbus";
    }
};

/// Ground height lookup for bolt endpoints.
pub const Ground = struct {
    heights: []const f32,
    map_size: usize,
    size: f32,
    height: f32,
    water_level: f32,

    /// World-space ground height at (x, z), bilinearly interpolated; 0 (sea level) outside the terrain.
    pub fn heightAt(self: Ground, x: f32, z: f32) f32 {
        const n: f32 = @floatFromInt(self.map_size - 1);
        const u = (x / self.size + 0.5) * n;
        const v = (z / self.size + 0.5) * n;
        if (u < 0 or v < 0 or u > n or v > n) return 0;
        const i_f = @min(@floor(u), n - 1);
        const j_f = @min(@floor(v), n - 1);
        const fu = u - i_f;
        const fv = v - j_f;
        const i: usize = @intFromFloat(i_f);
        const j: usize = @intFromFloat(j_f);
        const h00 = self.sample(i, j);
        const h10 = self.sample(i + 1, j);
        const h01 = self.sample(i, j + 1);
        const h11 = self.sample(i + 1, j + 1);
        const h = std.math.lerp(std.math.lerp(h00, h10, fu), std.math.lerp(h01, h11, fu), fv);
        return h * self.height;
    }

    fn sample(self: Ground, i: usize, j: usize) f32 {
        return @max(self.heights[j * self.map_size + i], self.water_level);
    }
};

pub const Lightning = struct {
    rng: std.Random.DefaultPrng,
    next_in: f32 = 2,
    age: f32 = std.math.inf(f32),
    stroke_times: [4]f32 = undefined,
    stroke_count: usize = 0,
    /// Peak brightness of this event (intra-cloud flashes are dimmer).
    strength: f32 = 0,
    /// World-space center of the in-cloud flash.
    flash_pos: rl.Vector3 = .{ .x = 0, .y = 0, .z = 0 },
    /// Bolt as segment endpoint pairs; empty for intra-cloud flashes.
    segments: [max_bolt_segments * 2]rl.Vector3 = undefined,
    segment_count: usize = 0,

    pub fn init(seed: u64) Lightning {
        return .{ .rng = std.Random.DefaultPrng.init(seed) };
    }

    /// Advance time; may spawn a new strike. `water` is cloud water content in 0..1.
    pub fn update(self: *Lightning, dt: f32, water: f32, params: cloudfield.Params, ground: Ground) void {
        self.age += dt;
        if (water < cloudfield.lightning_min_water) return;

        self.next_in -= dt;
        if (self.next_in > 0) return;

        // Storminess 0..1 sets the mean time between attempts (8 s down to 0.8 s).
        const storm = std.math.clamp((water - cloudfield.lightning_min_water) / (1 - cloudfield.lightning_min_water), 0, 1);
        const mean_gap = std.math.lerp(8.0, 0.8, storm);
        self.next_in = -@log(1 - self.rng.random().float(f32)) * mean_gap;

        self.strikeAtWettest(params, ground, false);
    }

    /// Search random columns for the wettest one and strike there if it is wet enough.
    /// `force` strikes a ground bolt at the wettest column regardless of thresholds.
    pub fn strikeAtWettest(self: *Lightning, params: cloudfield.Params, ground: Ground, force: bool) void {
        const r = self.rng.random();
        var best_water: f32 = -1;
        var best = [2]f32{ 0, 0 };
        for (0..strike_search_samples) |_| {
            const x = (r.float(f32) * 2 - 1) * strike_search_radius;
            const z = (r.float(f32) * 2 - 1) * strike_search_radius;
            const cw = cloudfield.columnWater(x * world_scale, z * world_scale, params);
            if (cw > best_water) {
                best_water = cw;
                best = .{ x, z };
            }
        }
        if (!force and best_water < params.lightning_threshold) return;

        // Wetter cells favor cloud-to-ground bolts; the rest flash inside the cloud.
        const excess = (best_water - params.lightning_threshold) / 0.05;
        const ground_bolt = force or r.float(f32) < std.math.clamp(0.3 + 0.4 * excess, 0.3, 0.7);
        self.start(best[0], best[1], ground_bolt, params.bottom / world_scale, ground);
    }

    fn start(self: *Lightning, x: f32, z: f32, ground_bolt: bool, base: f32, ground: Ground) void {
        const r = self.rng.random();

        self.age = 0;
        self.strength = if (ground_bolt) 1.0 else 0.4 + 0.3 * r.float(f32);
        self.flash_pos = .{ .x = x, .y = base + 4 + 6 * r.float(f32), .z = z };

        // 1-4 return strokes, 40-120 ms apart.
        self.stroke_count = 1 + r.uintLessThan(usize, 4);
        var st: f32 = 0;
        for (0..self.stroke_count) |i| {
            self.stroke_times[i] = st;
            st += 0.04 + 0.08 * r.float(f32);
        }

        self.segment_count = 0;
        if (!ground_bolt) return;

        const top = rl.Vector3.init(x, base, z);
        const bottom = rl.Vector3.init(x + (r.float(f32) - 0.5) * 12, ground.heightAt(x, z), z + (r.float(f32) - 0.5) * 12);
        var channel: [main_channel_points]rl.Vector3 = undefined;
        self.jaggedPath(&channel, top, bottom, true);
        for (0..main_channel_points - 1) |i| self.addSegment(channel[i], channel[i + 1]);

        // Two branches forking off the upper half of the channel, fading out in mid-air.
        for (0..2) |_| {
            const from = channel[3 + r.uintLessThan(usize, main_channel_points / 2)];
            const len = 6 + 8 * r.float(f32);
            const to = from.add(.{
                .x = (r.float(f32) - 0.5) * 2 * len,
                .y = -len,
                .z = (r.float(f32) - 0.5) * 2 * len,
            });
            var branch: [max_branch_segments + 1]rl.Vector3 = undefined;
            self.jaggedPath(&branch, from, to, false);
            for (0..max_branch_segments) |i| self.addSegment(branch[i], branch[i + 1]);
        }
    }

    /// Random walk from a toward b; if `pin_end`, the drift is corrected so the path ends at b.
    fn jaggedPath(self: *Lightning, out: []rl.Vector3, a: rl.Vector3, b: rl.Vector3, pin_end: bool) void {
        const r = self.rng.random();
        const n = out.len;
        const step = a.distance(b) / @as(f32, @floatFromInt(n - 1));
        var dx: f32 = 0;
        var dz: f32 = 0;
        for (out, 0..) |*p, i| {
            const k = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n - 1));
            p.* = a.lerp(b, k).add(.{ .x = dx, .y = 0, .z = dz });
            dx += (r.float(f32) - 0.5) * step * 1.2;
            dz += (r.float(f32) - 0.5) * step * 1.2;
        }
        if (pin_end) {
            const end_off = out[n - 1].subtract(b);
            for (out, 0..) |*p, i| {
                const k = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n - 1));
                p.* = p.subtract(end_off.scale(k));
            }
        }
    }

    fn addSegment(self: *Lightning, a: rl.Vector3, b: rl.Vector3) void {
        if (self.segment_count >= max_bolt_segments) return;
        self.segments[self.segment_count * 2] = a;
        self.segments[self.segment_count * 2 + 1] = b;
        self.segment_count += 1;
    }

    /// Current brightness: a sharp decaying pulse per return stroke.
    pub fn intensity(self: Lightning) f32 {
        var sum: f32 = 0;
        for (self.stroke_times[0..self.stroke_count]) |st| {
            const a = self.age - st;
            if (a >= 0) sum += @exp(-a / 0.045);
        }
        return @min(sum, 1.5) * self.strength;
    }

    /// Bolt segments to draw this frame (empty once the flash has faded).
    pub fn visibleSegments(self: Lightning) []const rl.Vector3 {
        if (self.intensity() < 0.01) return &.{};
        return self.segments[0 .. self.segment_count * 2];
    }
};

/// A single tornado: spawns under the wettest cell of a supercell storm, drifts with the wind,
/// lowers its funnel to touch down, then ropes out and lifts.
pub const Tornado = struct {
    rng: std.Random.DefaultPrng,
    active: bool = false,
    /// Ground position, world units.
    pos: [2]f32 = .{ 0, 0 },
    /// Ground velocity, world units per second.
    vel: [2]f32 = .{ 0, 0 },
    age: f32 = 0,
    lifetime: f32 = 0,
    wander_phase: f32 = 0,
    next_check: f32 = 10,

    pub const min_water: f32 = 0.75;
    pub const min_cloud_type: f32 = 0.75;
    const form_time: f32 = 12;
    const decay_time: f32 = 15;

    pub fn init(seed: u64) Tornado {
        return .{ .rng = std.Random.DefaultPrng.init(seed) };
    }

    pub fn update(self: *Tornado, dt: f32, sky: Weather, params: cloudfield.Params) void {
        const r = self.rng.random();
        if (self.active) {
            self.age += dt;
            if (self.age >= self.lifetime) self.active = false;

            // Carried by the storm's steering wind, plus a slow sideways wander.
            const w = sky.wind();
            const dir = rl.Vector2.init(w[0], w[1]).normalize();
            self.wander_phase += dt * 0.15;
            const wander = @sin(self.wander_phase) * 0.12;
            self.vel = .{
                w[0] / world_scale * 0.7 - dir.y * wander,
                w[1] / world_scale * 0.7 + dir.x * wander,
            };
            self.pos[0] += self.vel[0] * dt;
            self.pos[1] += self.vel[1] * dt;
            return;
        }

        self.next_check -= dt;
        if (self.next_check > 0) return;
        self.next_check = 10;
        if (sky.water < min_water or sky.cloud_type < min_cloud_type) return;

        const severity = std.math.clamp((sky.water - min_water) / (1 - min_water), 0, 1);
        if (r.float(f32) > 0.15 + 0.35 * severity) return;

        // Form under the wettest column in range, if it is storm-grade.
        var best_water: f32 = -1;
        var best = [2]f32{ 0, 0 };
        for (0..16) |_| {
            const x = (r.float(f32) * 2 - 1) * 100;
            const z = (r.float(f32) * 2 - 1) * 100;
            const cw = cloudfield.columnWater(x * world_scale, z * world_scale, params);
            if (cw > best_water) {
                best_water = cw;
                best = .{ x, z };
            }
        }
        if (best_water < params.lightning_threshold) return;
        self.spawnAt(best[0], best[1]);
    }

    pub fn spawnAt(self: *Tornado, x: f32, z: f32) void {
        const r = self.rng.random();
        self.active = true;
        self.pos = .{ x, z };
        self.vel = .{ 0, 0 };
        self.age = 0;
        self.lifetime = 60 + 120 * r.float(f32);
        self.wander_phase = r.float(f32) * std.math.tau;
    }

    /// Overall presence 0..1: grows while forming, fades (ropes out) at the end.
    pub fn strength(self: Tornado) f32 {
        if (!self.active) return 0;
        const grow = std.math.clamp(self.age / form_time, 0, 1);
        const fade = std.math.clamp((self.lifetime - self.age) / decay_time, 0, 1);
        return @min(grow, fade);
    }

    /// Parent storm cell strength: organizes ahead of the funnel and lingers after it.
    pub fn cellStrength(self: Tornado) f32 {
        if (!self.active) return 0;
        const grow = std.math.clamp(self.age / 5, 0, 1);
        const fade = std.math.clamp((self.lifetime - self.age) / decay_time, 0, 1);
        return @min(grow, @max(fade, 0.3));
    }

    /// Storm cell center in meters: above the funnel top.
    pub fn cellPos(self: Tornado) [2]f32 {
        const l = self.lean();
        return .{ (self.pos[0] + l[0]) * world_scale, (self.pos[1] + l[1]) * world_scale };
    }

    /// How far the funnel reaches down from the cloud base, 0..1 (1 = on the ground).
    pub fn touchdown(self: Tornado) f32 {
        if (!self.active) return 0;
        const lower = std.math.clamp(self.age / form_time, 0, 1);
        const lift = std.math.clamp((self.lifetime - self.age) / (decay_time * 0.8), 0, 1);
        return @min(lower, lift);
    }

    /// Offset of the funnel top relative to its base, world units: the parent cloud lags behind
    /// the ground circulation, plus a slow sway.
    pub fn lean(self: Tornado) [2]f32 {
        const sway = @sin(self.age * 0.3) * 3;
        return .{ -self.vel[0] * 60 + sway, -self.vel[1] * 60 + @cos(self.age * 0.23) * 3 };
    }
};
