//! CPU mirror of the cloud density field in atmosphere.fs, so game logic (lightning placement)
//! can ask where the water is. Constants and noise must stay in sync with the shader.

const std = @import("std");

const V3 = @Vector(3, f32);

/// Cloud-field parameters, derived from water content (0..1) and cloud type
/// (0 = stratus, 0.5 = cumulus, 1 = cumulonimbus). Mirrored by uniforms in atmosphere.fs.
pub const Params = struct {
    coverage: f32,
    density_scale: f32,
    /// Column water above which a cell can produce lightning (> 1 disables lightning).
    lightning_threshold: f32,
    /// Layer base and top in meters above the surface.
    bottom: f32,
    top: f32,
    /// Horizontal feature size in meters.
    scale: f32,
    /// Height fraction where tops start rounding off (high = flat-topped sheets).
    shape_top: f32,
    /// Accumulated wind drift of the cloud field, meters.
    offset: [2]f32,
    /// Unit wind direction (x, z).
    wind_dir: [2]f32,
    /// Parent storm cell (mesocyclone) center in meters and strength 0..1; boosts coverage around it.
    cell_pos: [2]f32 = .{ 0, 0 },
    cell_strength: f32 = 0,

    pub fn init(water: f32, cloud_type: f32, offset: [2]f32, wind_dir: [2]f32) Params {
        const w = water;
        // Stratus spreads into near-continuous sheets.
        const sheet_bonus = 0.08 * (1 - smoothstep(0, 0.5, cloud_type));
        return .{
            .coverage = std.math.lerp(0.45, 0.75, w) + sheet_bonus,
            .density_scale = std.math.lerp(0.7, 2.5, w),
            // Calibrated against columnWater's distribution: lightning needs the top ~1% of columns
            // at w = 0.6, ~15% at w = 1.
            .lightning_threshold = if (w < lightning_min_water) 2.0 else std.math.lerp(0.29, 0.25, w),
            .bottom = typeBlend(cloud_type, 3300, 3500, 3200),
            .top = typeBlend(cloud_type, 4100, 5000, 6500),
            .scale = typeBlend(cloud_type, 9000, 4000, 5500),
            .shape_top = typeBlend(cloud_type, 0.85, 0.5, 0.3),
            .offset = offset,
            .wind_dir = wind_dir,
        };
    }
};

/// Piecewise-linear blend between stratus (t=0), cumulus (t=0.5) and cumulonimbus (t=1) values.
fn typeBlend(t: f32, stratus: f32, cumulus: f32, cumulonimbus: f32) f32 {
    if (t < 0.5) return std.math.lerp(stratus, cumulus, t * 2);
    return std.math.lerp(cumulus, cumulonimbus, (t - 0.5) * 2);
}

/// Storm-cell coverage boost; must match CELL_COVERAGE / CELL_R in atmosphere.fs.
pub const cell_coverage: f32 = 0.22;
pub const cell_radius: f32 = 5000.0;

pub const lightning_min_water: f32 = 0.6;

fn fract(x: f32) f32 {
    return x - @floor(x);
}

fn splat(x: f32) V3 {
    return @splat(x);
}

fn hash(p: V3) f32 {
    var q = p * splat(0.3183099) + splat(0.1);
    q = (q - @floor(q)) * splat(17.0);
    return fract(q[0] * q[1] * q[2] * (q[0] + q[1] + q[2]));
}

fn mix(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

fn valueNoise(x: V3) f32 {
    const i = @floor(x);
    var f = x - i;
    f = f * f * (splat(3.0) - splat(2.0) * f);
    return mix(
        mix(mix(hash(i + V3{ 0, 0, 0 }), hash(i + V3{ 1, 0, 0 }), f[0]), mix(hash(i + V3{ 0, 1, 0 }), hash(i + V3{ 1, 1, 0 }), f[0]), f[1]),
        mix(mix(hash(i + V3{ 0, 0, 1 }), hash(i + V3{ 1, 0, 1 }), f[0]), mix(hash(i + V3{ 0, 1, 1 }), hash(i + V3{ 1, 1, 1 }), f[0]), f[1]),
        f[2],
    );
}

fn fbm(p0: V3, octaves: u32) f32 {
    var p = p0;
    var sum: f32 = 0;
    var amp: f32 = 0.5;
    var norm: f32 = 0;
    for (0..octaves) |_| {
        sum += amp * valueNoise(p);
        norm += amp;
        amp *= 0.5;
        p = p * splat(2.03) + V3{ 17.1, 3.7, 9.2 };
    }
    return sum / norm;
}

fn smoothstep(e0: f32, e1: f32, x: f32) f32 {
    const t = std.math.clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

/// Average normalized cloud water in the vertical column at (x, z) meters.
/// Mirrors `columnWater` in atmosphere.fs.
pub fn columnWater(x: f32, z: f32, p: Params) f32 {
    const dx = x - p.cell_pos[0];
    const dz = z - p.cell_pos[1];
    const coverage = p.coverage + cell_coverage * p.cell_strength * @exp(-(dx * dx + dz * dz) / (cell_radius * cell_radius));
    const n = 4;
    var sum: f32 = 0;
    for (0..n) |i| {
        const h = (@as(f32, @floatFromInt(i)) + 0.5) / n;
        const alt = p.bottom + h * (p.top - p.bottom);
        const q = V3{ x - p.offset[0], alt, z - p.offset[1] } / splat(p.scale);
        const shape = smoothstep(0.0, 0.1, h) * smoothstep(1.0, p.shape_top, h);
        const d = fbm(q, 3) * shape - (1.0 - coverage);
        if (d > 0) sum += smoothstep(0.0, 0.25, d) * d;
    }
    return sum / n;
}
