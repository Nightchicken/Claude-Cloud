#version 330

// Full-screen atmosphere pass: single-scattering Rayleigh + Mie, a ray-marched cloud layer,
// lightning and tornadoes, all driven by cloud water content (see cloudfield.zig).
// Sky pixels integrate to the edge of the atmosphere; terrain pixels integrate
// up to the depth-buffer distance (aerial perspective).

in vec2 fragTexCoord;

uniform sampler2D texture0; // scene color
uniform sampler2D depthTex; // scene depth
uniform vec3 camPos;
uniform vec3 camForward;
uniform vec3 camRight;
uniform vec3 camUp;
uniform float tanHalfFov;
uniform float aspect;
uniform vec3 sunDir;
uniform float time;

// Weather, derived from water content on the CPU (cloudfield.Params).
uniform float cloudCoverage;
uniform float cloudDensityScale;
uniform float dropletDiameter; // micrometers, 5-50
uniform float cloudBottom; // layer base, meters above the surface
uniform float cloudTop;
uniform float cloudScale; // horizontal feature size, meters
uniform float cloudShapeTop; // height fraction where tops start rounding (high = flat sheets)
uniform vec2 cloudOffset; // accumulated wind drift, meters
uniform vec2 windDir; // unit wind direction (x, z)

// Lightning: flash light inside the cloud, and the bolt as segment endpoint pairs (world units).
const int BOLT_MAX_SEGMENTS = 40; // must match weather.max_bolt_segments
uniform vec3 flashPos;
uniform float flashIntensity;
uniform vec3 boltSegs[BOLT_MAX_SEGMENTS * 2];
uniform int boltSegCount;

// Tornado (world units); strength 0 disables it.
uniform vec3 tornadoPos; // base on the ground
uniform vec2 tornadoLean; // top offset relative to base
uniform float tornadoStrength; // 0..1, ropes out as it fades
uniform float tornadoTouchdown; // 0..1, how far the funnel reaches down from the cloud base

// Parent storm cell (mesocyclone) anchored above the tornado: denser, rotating cloud with a lowered
// wall cloud. Mirrored (coverage only) by cloudfield.columnWater.
uniform vec2 cellPos; // local meters (x, z)
uniform float cellStrength; // 0..1
uniform float cellAge; // seconds since the cell formed; drives its rotation

out vec4 finalColor;

const float PI = 3.14159265;
const float NEAR = 0.05; // raylib RL_CULL_DISTANCE_NEAR
const float FAR = 4000.0; // raylib RL_CULL_DISTANCE_FAR
const float WORLD_SCALE = 100.0; // meters per world unit

const float R_PLANET = 6371e3;
const float R_ATMOS = 6471e3;
const vec3 BETA_R = vec3(5.5e-6, 13.0e-6, 22.4e-6);
const float BETA_M = 40e-6; // aerosol haze; makes light shafts visible
const float H_R = 8e3;
const float H_M = 1.2e3;
const float MIE_G = 0.76;
const float SUN_INTENSITY = 22.0;
const float SURFACE_LIGHT = 2.0;

const int SHAFT_STEPS = 20; // view samples within SHAFT_DIST, with cloud shadows
const int FAR_STEPS = 8; // view samples beyond SHAFT_DIST
const float SHAFT_DIST = 25000.0;
const int LIGHT_STEPS = 8;

// Cloud layer
const float CLOUD_DENSITY = 0.03; // extinction per meter at full density (scaled by cloudDensityScale)
const float CLOUD_DETAIL_SCALE = 6.0; // detail noise frequency relative to base
const float CLOUD_EROSION = 0.18; // how much detail noise eats into cloud edges
const float CLOUD_MAX_DIST = 30000.0;
const int CLOUD_STEPS = 64;
const int CLOUD_LIGHT_STEPS = 6;
const float CLOUD_LIGHT_STEP = 50.0; // first light step; each next one doubles (~3 km total)
const float CLOUD_SUN = 30.0;
const float CLOUD_AMBIENT = 0.45;

// Storm cell (must match cloudfield.cell_coverage / cell_radius for coverage)
const float CELL_R = 5000.0; // radius of the coverage boost, meters
const float CELL_COVERAGE = 0.22;
const float WALL_R = 1600.0; // wall cloud radius
const float WALL_DROP = 450.0; // how far the wall cloud lowers below the base
const float CELL_SPIN = 0.05; // rotation rate at the center, rad/s
const float WALL_DENSITY = 0.022; // guaranteed cloud in the wall cloud the funnel hangs from
const float WALL_THICKNESS = 500.0; // how far the wall cloud extends above the normal base

// Lightning
const vec3 FLASH_COLOR = vec3(0.75, 0.8, 1.0);
const float FLASH_RADIUS = 900.0; // light falloff inside the cloud, meters
const float FLASH_CLOUD = 6.0;
const float FLASH_GROUND = 1.5;
const float FLASH_GROUND_RADIUS = 6000.0;
const float BOLT_WIDTH = 3.0; // core radius in meters (widened to at least a pixel)
const float BOLT_BRIGHTNESS = 25.0;

// Tornado
const float TORNADO_R_GROUND = 60.0; // funnel radius at the ground, meters
const float TORNADO_R_TOP = 380.0; // radius just below the wall cloud
const float TORNADO_WALL_R = 150.0; // small flare where it joins the wall cloud
const float TORNADO_DENSITY = 0.035; // peak extinction per meter in the funnel wall
const float DEBRIS_DENSITY = 0.01;
const vec3 DEBRIS_ALBEDO = vec3(0.42, 0.34, 0.25); // dust absorbs far more than droplets
const float DEBRIS_G = 0.45; // broad forward-scattering lobe for dust
const float TORNADO_STEP = 8.0; // sample spacing near the funnel, meters (grows with distance)
const int MEDIUM_MAX_ITER = 320; // unified cloud + tornado march: cloud steps + funnel steps + skips

// Multiple-scattering octaves: each octave scales extinction by A, contribution by B, phase eccentricity by C.
const int MS_OCTAVES = 4;
const float MS_A = 0.35;
const float MS_B = 0.5;
const float MS_C = 0.5;

vec2 raySphere(vec3 ro, vec3 rd, float r) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float d = b * b - c;
    if (d < 0.0) return vec2(-1.0); // miss: .x > 0 checks must not see a hit
    d = sqrt(d);
    return vec2(-b - d, -b + d);
}

// Optical depth (Rayleigh, Mie) from p toward the sun; negative if the planet blocks it.
vec2 sunOpticalDepth(vec3 p) {
    if (raySphere(p, sunDir, R_PLANET).x > 0.0) return vec2(-1.0);
    float ds = raySphere(p, sunDir, R_ATMOS).y / float(LIGHT_STEPS);
    vec2 od = vec2(0.0);
    for (int i = 0; i < LIGHT_STEPS; i++) {
        float h = length(p + sunDir * (float(i) + 0.5) * ds) - R_PLANET;
        od += vec2(exp(-h / H_R), exp(-h / H_M)) * ds;
    }
    return od;
}

vec3 extinction(vec2 od) {
    return exp(-(BETA_R * od.x + BETA_M * 1.1 * od.y));
}

// Rough sky light, fading out as the sun sets.
vec3 skyAmbient() {
    return vec3(0.78, 0.83, 0.93) * (0.15 + 0.85 * clamp(sunDir.y * 4.0, 0.0, 1.0));
}

vec3 flashP() {
    return flashPos * WORLD_SCALE + vec3(0.0, R_PLANET, 0.0);
}

float cloudShadow(vec3 p);

// Interleaved gradient noise, shifted every frame: fine-grained jitter that hides step banding.
float ign(vec2 p) {
    p += 5.588238 * floor(time * 60.0);
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// `weather` adds cloud shadows (light shafts); only used for the near segment.
void scatterSegment(vec3 ro, vec3 rd, float t0, float t1, int steps, bool quadratic, bool weather, float jitter,
                    inout vec2 od, inout vec3 sumR, inout vec3 sumM) {
    for (int i = 0; i < steps; i++) {
        float a0 = float(i) / float(steps);
        float a1 = float(i + 1) / float(steps);
        if (quadratic) {
            a0 *= a0;
            a1 *= a1;
        }
        float ta = mix(t0, t1, a0);
        float tb = mix(t0, t1, a1);
        vec3 p = ro + rd * mix(ta, tb, jitter);
        float h = length(p) - R_PLANET;
        vec2 d = vec2(exp(-h / H_R), exp(-h / H_M)) * (tb - ta);
        od += d;

        vec2 sod = sunOpticalDepth(p);
        vec3 attn = sod.x < 0.0 ? vec3(0.0) : extinction(od + sod);
        if (weather) attn *= cloudShadow(p);
        sumR += d.x * attn;
        sumM += d.y * attn;

    }
}

void scatter(vec3 ro, vec3 rd, float tMax, out vec3 inscatter, out vec3 transmittance) {
    vec2 od = vec2(0.0);
    vec3 sumR = vec3(0.0);
    vec3 sumM = vec3(0.0);
    float jitter = ign(gl_FragCoord.xy);

    // Dense near segment with cloud shadows (light shafts), then a sparse far segment without.
    float near = min(tMax, SHAFT_DIST);
    scatterSegment(ro, rd, 0.0, near, SHAFT_STEPS, false, true, jitter, od, sumR, sumM);
    if (tMax > near) scatterSegment(ro, rd, near, tMax, FAR_STEPS, true, false, jitter, od, sumR, sumM);

    float mu = dot(rd, sunDir);
    float g2 = MIE_G * MIE_G;
    float phaseR = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    float phaseM = 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu)) /
        ((2.0 + g2) * pow(1.0 + g2 - 2.0 * MIE_G * mu, 1.5));

    inscatter = SUN_INTENSITY * (sumR * BETA_R * phaseR + sumM * BETA_M * phaseM);
    transmittance = extinction(od);
}

float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float valueNoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash(i + vec3(0, 0, 0)), hash(i + vec3(1, 0, 0)), f.x),
            mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
        mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
            mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y),
        f.z);
}

float fbm(vec3 p, int octaves) {
    float sum = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    for (int i = 0; i < octaves; i++) {
        sum += amp * valueNoise(p);
        norm += amp;
        amp *= 0.5;
        p = p * 2.03 + vec3(17.1, 3.7, 9.2);
    }
    return sum / norm;
}

// Storm cell influence at local xz for a given radius, 0..cellStrength.
float cellFalloff(vec2 xz, float radius) {
    vec2 d = xz - cellPos;
    return cellStrength * exp(-dot(d, d) / (radius * radius));
}

// Lowest point of the cloud layer anywhere (the wall cloud dips below the base).
float layerBottom() {
    return cloudBottom - WALL_DROP * cellStrength;
}

// `detail` adds high-frequency erosion; light/shadow marches skip it for speed.
float cloudDensity(vec3 p, bool detail) {
    vec3 local = p - vec3(0.0, R_PLANET, 0.0);
    float alt = length(p) - R_PLANET;
    float bottom = cloudBottom - WALL_DROP * cellFalloff(local.xz, WALL_R);
    float h = (alt - bottom) / (cloudTop - bottom);
    if (h < 0.0 || h > 1.0) return 0.0;

    float coverage = cloudCoverage;
    float wall = 0.0;
    if (cellStrength > 0.0) {
        // Mesocyclone: the cloud around the tornado rotates as a body (faster near the center)
        // and stays anchored to the storm instead of drifting off with the wind.
        vec2 rel = local.xz - cellPos;
        float f = exp(-dot(rel, rel) / (CELL_R * CELL_R));
        float a = cellAge * CELL_SPIN * f;
        float ca = cos(a);
        float sa = sin(a);
        local.xz = cellPos + vec2(ca * rel.x - sa * rel.y, sa * rel.x + ca * rel.y);
        coverage += CELL_COVERAGE * cellStrength * f;

        // Wall cloud: a guaranteed, rotating lowered block of cloud centered on the funnel top, so
        // the funnel always flows into solid cloud instead of ending in a gap of the noise field.
        float w = exp(-1.5 * dot(rel, rel) / (WALL_R * WALL_R));
        if (w > 0.01 && alt < cloudBottom + WALL_THICKNESS) {
            vec3 wq = vec3(local.x, alt * 2.0, local.z) / 600.0;
            float n = detail ? fbm(wq, 3) : valueNoise(wq);
            float vert = smoothstep(bottom, bottom + 150.0, alt) * (1.0 - smoothstep(cloudBottom, cloudBottom + WALL_THICKNESS, alt));
            wall = cellStrength * w * vert * (0.5 + n) * WALL_DENSITY;
        }
    }
    local.xz -= cloudOffset;
    vec3 q = local / cloudScale;

    // Flat-ish bottoms, rounded tops.
    float shape = smoothstep(0.0, 0.1, h) * smoothstep(1.0, cloudShapeTop, h);
    float d = fbm(q, 4) * shape - (1.0 - coverage);
    if (d <= 0.0) return wall;

    if (detail) {
        // Erode mostly near the edges (low density) so cores stay intact but borders go wispy.
        float n = fbm(q * CLOUD_DETAIL_SCALE + vec3(0.0, time * 0.01, 0.0), 3);
        d -= n * CLOUD_EROSION * (1.0 - smoothstep(0.0, 0.2, d));
        if (d <= 0.0) return wall;
    }
    // Soft ramp from the edge instead of a hard threshold.
    return smoothstep(0.0, 0.25, d) * d * CLOUD_DENSITY * cloudDensityScale + wall;
}

// Average normalized cloud water in the vertical column at local (x, z) meters.
// Mirrors cloudfield.columnWater, which places lightning.
float columnWater(vec2 xz) {
    float coverage = cloudCoverage + CELL_COVERAGE * cellFalloff(xz, CELL_R);
    float sum = 0.0;
    for (int i = 0; i < 4; i++) {
        float h = (float(i) + 0.5) / 4.0;
        float alt = cloudBottom + h * (cloudTop - cloudBottom);
        vec3 q = vec3(xz.x - cloudOffset.x, alt, xz.y - cloudOffset.y) / cloudScale;
        float shape = smoothstep(0.0, 0.1, h) * smoothstep(1.0, cloudShapeTop, h);
        float d = fbm(q, 3) * shape - (1.0 - coverage);
        if (d > 0.0) sum += smoothstep(0.0, 0.25, d) * d;
    }
    return sum / 4.0;
}

// Fraction of sunlight reaching p through the cloud layer.
float cloudShadow(vec3 p) {
    float alt = length(p) - R_PLANET;
    if (alt > cloudTop) return 1.0;
    float t0 = alt < layerBottom() ? raySphere(p, sunDir, R_PLANET + layerBottom()).y : 0.0;
    float t1 = raySphere(p, sunDir, R_PLANET + cloudTop).y;
    t1 = min(t1, t0 + 8000.0);
    if (t1 <= t0) return 1.0;
    const int N = 4;
    float dt = (t1 - t0) / float(N);
    float od = 0.0;
    for (int i = 0; i < N; i++) {
        od += cloudDensity(p + sunDir * (t0 + (float(i) + 0.5) * dt), false) * dt;
    }
    // Shadows are softened by light scattered around the cloud, so don't let them go fully black.
    return mix(0.08, 1.0, exp(-od));
}

float hgPhase(float mu, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}

float drainePhase(float mu, float g, float alpha) {
    float g2 = g * g;
    return hgPhase(mu, g) * (1.0 + alpha * mu * mu) / (1.0 + alpha * (1.0 + 2.0 * g2) / 3.0);
}

// Jendersie & d'Eon 2023, "An Approximate Mie Scattering Function for Fog and Cloud Rendering".
// An HG lobe for the sharp forward diffraction peak, blended with a Draine lobe for the
// wider refraction/reflection scattering. `ecc` scales both lobes' eccentricity (for MS octaves).
float miePhase(float mu, float d, float ecc) {
    float gHG = exp(-0.0990567 / (d - 1.67154));
    float gD = exp(-2.20679 / (d + 3.91029) - 0.428934);
    float alpha = exp(3.62489 - 8.29288 / (d + 5.52825));
    float wD = exp(-0.599085 / (d - 0.641583) - 0.665888);
    return mix(hgPhase(mu, gHG * ecc), drainePhase(mu, gD * ecc, alpha), wD);
}

// Funnel centerline at height fraction hf (0 = ground, 1 = cloud base), local meters (xz).
vec2 funnelAxis(float hf) {
    vec2 base = tornadoPos.xz * WORLD_SCALE;
    vec2 lean = tornadoLean * WORLD_SCALE * hf * hf;
    float snake = sin(hf * 7.0 + time * 0.8) * 45.0 * hf * (1.0 - hf) * 4.0;
    return base + lean + vec2(snake, snake * 0.6);
}

// Funnel (condensation) and debris extinction at local point p, already scaled by strength.
void tornadoDensity(vec3 p, float baseY, float topY, out float funnelS, out float debrisS) {
    funnelS = 0.0;
    debrisS = 0.0;
    float hf = (p.y - baseY) / (topY - baseY);
    if (hf < -0.05 || hf > 1.0) return;
    float hc = clamp(hf, 0.0, 1.0);

    vec2 rel = p.xz - funnelAxis(hc);
    float d = length(rel);
    float ang = atan(rel.y, rel.x);

    // Funnel radius: narrow at the ground, widening upward, flaring into the wall cloud;
    // thins as it ropes out (low strength).
    float r = mix(TORNADO_R_GROUND, TORNADO_R_TOP, pow(hc, 2.5)) * (0.35 + 0.65 * tornadoStrength);
    r += smoothstep(0.8, 1.0, hc) * TORNADO_WALL_R;
    float x = d / r;

    if (x < 1.6) {
        // Spiral condensation bands: the angle is twisted with height and spun faster near the ground.
        float spin = time * (1.2 + 3.0 * (1.0 - hc));
        float a2 = ang - spin + hc * 6.0;
        float n = fbm(vec3(cos(a2) * 2.0, sin(a2) * 2.0, p.y / 300.0 - time * 0.4), 3);

        // Sheath profile: a coherent condensation tube, densest in the rotating wall, slightly thinner
        // core, with a soft edge only lightly broken up; bands modulate rather than shred it.
        // The edge is displaced by smooth low-frequency noise (fine value noise shows its lattice as teeth).
        float edgeN = valueNoise(vec3(cos(a2) * 1.5, sin(a2) * 1.5, p.y / 350.0 - time * 0.3));
        float edge = 1.0 - smoothstep(0.6, 1.0, x + (edgeN - 0.5) * 0.3);
        float core = mix(0.65, 1.0, smoothstep(0.1, 0.6, x));
        float bands = 0.85 + 0.3 * n;
        float reach = smoothstep(1.0 - tornadoTouchdown, 1.0 - tornadoTouchdown + 0.06, hf);
        // Crossfade into the wall cloud over the top of the funnel (no hard cut at topY).
        float merge = 1.0 - smoothstep(0.8, 1.0, hf);
        funnelS = edge * core * bands * reach * merge * mix(0.75, 1.0, hc) * TORNADO_DENSITY;
    }

    // Debris cloud swirling around the base once it is on the ground.
    float dy = p.y - baseY;
    float ground = smoothstep(0.9, 1.0, tornadoTouchdown);
    if (ground > 0.0 && dy < 600.0 && d < TORNADO_R_GROUND * 8.0) {
        float da = ang - time * 3.0;
        float dn = fbm(vec3(cos(da) * 3.0, sin(da) * 3.0, dy / 60.0 - time * 0.3), 3);
        float dr = d / TORNADO_R_GROUND;
        debrisS = ground * exp(-max(dy, 0.0) / 160.0) * (1.0 - smoothstep(2.0, 6.5, dr + (dn - 0.5) * 3.0)) *
            (0.3 + 1.2 * dn) * DEBRIS_DENSITY;
    }

    funnelS *= tornadoStrength;
    debrisS *= tornadoStrength;
}

// Conservative horizontal distance (meters) from p to the outside of the funnel and debris volumes;
// <= 0 means p may be inside. Lets the march skip empty space and sample finely only in the funnel.
float tornadoBound(vec3 p, float baseY, float topY) {
    float hf = clamp((p.y - baseY) / (topY - baseY), 0.0, 1.0);
    float r = mix(TORNADO_R_GROUND, TORNADO_R_TOP, pow(hf, 2.5)) * (0.35 + 0.65 * tornadoStrength);
    r += smoothstep(0.8, 1.0, hf) * TORNADO_WALL_R;
    float funnel = length(p.xz - funnelAxis(hf)) - (r * 1.3 + 60.0);
    funnel = max(funnel, max(baseY - 30.0 - p.y, p.y - topY));
    float debris = 1e9;
    if (tornadoTouchdown > 0.9) {
        debris = length(p.xz - tornadoPos.xz * WORLD_SCALE) - TORNADO_R_GROUND * 7.5;
        debris = max(debris, p.y - baseY - 600.0);
    }
    return min(funnel, debris);
}

// Ray interval [t0, t1] through the tornado's bounding cylinder (local frame), clipped to maxDist.
bool tornadoInterval(vec3 o, vec3 rd, float maxDist, float baseY, float topY, out float t0, out float t1) {
    vec2 c = tornadoPos.xz * WORLD_SCALE + tornadoLean * WORLD_SCALE * 0.5;
    float R = TORNADO_R_TOP + TORNADO_WALL_R + length(tornadoLean) * WORLD_SCALE * 0.5 + 100.0;
    vec2 oc = o.xz - c;
    float a = dot(rd.xz, rd.xz);
    t0 = 0.0;
    t1 = maxDist;
    if (a > 1e-8) {
        float b = dot(oc, rd.xz);
        float disc = b * b - a * (dot(oc, oc) - R * R);
        if (disc < 0.0) return false;
        disc = sqrt(disc);
        t0 = max(t0, (-b - disc) / a);
        t1 = min(t1, (-b + disc) / a);
    } else if (dot(oc, oc) > R * R) {
        return false;
    }
    if (abs(rd.y) > 1e-6) {
        float ya = (baseY - 30.0 - o.y) / rd.y;
        float yb = (topY - o.y) / rd.y;
        t0 = max(t0, min(ya, yb));
        t1 = min(t1, max(ya, yb));
    } else if (o.y < baseY - 30.0 || o.y > topY) {
        return false;
    }
    return t1 > t0;
}

// Ray-march the whole cloud medium up to tLimit: the cloud layer and the tornado (funnel + debris)
// are one participating medium. Densities add at every sample and every sample is lit the same
// way (one light march through all of it, Mie phase + multiple-scattering octaves, deck shadowing,
// sky ambient), so the funnel blends into the wall cloud with matching shading.
// Returns in-scattered light, transmittance and mean hit depth.
void marchClouds(vec3 ro, vec3 rd, float tLimit, out vec3 light, out float trans, out float depth) {
    light = vec3(0.0);
    trans = 1.0;
    depth = 0.0;

    // Cloud layer interval.
    vec2 top = raySphere(ro, rd, R_PLANET + cloudTop);
    vec2 bot = raySphere(ro, rd, R_PLANET + layerBottom());
    float alt0 = length(ro) - R_PLANET;
    float c0 = 0.0;
    float c1 = -1.0;
    if (alt0 > cloudTop) {
        if (top.x > 0.0) {
            c0 = top.x;
            c1 = bot.x > 0.0 ? bot.x : top.y;
        }
    } else if (alt0 < layerBottom()) {
        c0 = bot.y;
        c1 = top.y;
    } else {
        c1 = bot.x > 0.0 ? bot.x : top.y;
    }
    c1 = min(min(c1, tLimit), c0 + CLOUD_MAX_DIST);
    bool clouds = c1 > c0;

    // Tornado interval, in the local frame (same t: the frames differ by a constant offset).
    vec3 planetOffset = vec3(0.0, R_PLANET, 0.0);
    vec3 o = ro - planetOffset;
    float baseY = tornadoPos.y * WORLD_SCALE;
    float topY = cloudBottom - WALL_DROP * cellStrength + 200.0; // reaches up into the wall cloud
    float k0 = 0.0;
    float k1 = -1.0;
    bool tornado = tornadoStrength > 0.0 && tornadoInterval(o, rd, tLimit, baseY, topY, k0, k1);

    if (!clouds && !tornado) return;
    float start = min(clouds ? c0 : 1e12, tornado ? k0 : 1e12);
    float end = max(clouds ? c1 : -1.0, tornado ? k1 : -1.0);

    vec3 ambient = skyAmbient() * CLOUD_AMBIENT;
    vec3 fp = flashP();

    float mu = dot(rd, sunDir);
    float phases[MS_OCTAVES];
    float ecc = 1.0;
    for (int n = 0; n < MS_OCTAVES; n++) {
        phases[n] = miePhase(mu, dropletDiameter, ecc);
        ecc *= MS_C;
    }
    float dustPhase = hgPhase(mu, DEBRIS_G);

    float cloudDs = clouds ? (c1 - c0) / float(CLOUD_STEPS) : 1e12;
    float jitter = ign(gl_FragCoord.xy + 17.0);
    float deckT = -1e12;
    float deckCached = 1.0;
    // Sunlight color is evaluated where each sample is (cached every ~300 m), not once per ray:
    // a per-ray value made rays that start low (through the tornado) differ from their neighbors.
    float sunT = -1e12;
    vec3 sunColor = vec3(0.0);
    float t = start;
    float weight = 0.0;
    for (int i = 0; i < MEDIUM_MAX_ITER; i++) {
        if (t >= end) break;
        vec3 p = ro + rd * t;
        bool inLayer = clouds && t >= c0 && t <= c1;
        bool inTornadoSpan = tornado && t >= k0 && t <= k1;
        float bound = inTornadoSpan ? tornadoBound(p - planetOffset, baseY, topY) : 1e12;
        bool nearFunnel = bound <= 0.0;

        if (!inLayer && !nearFunnel) {
            // Empty space: skip toward the funnel or the start of the next interval.
            float skip = inTornadoSpan ? max(bound * 0.7, 4.0) : 1e12;
            if (clouds && t < c0) skip = min(skip, c0 - t + 0.01);
            if (tornado && t < k0) skip = min(skip, k0 - t + 0.01);
            if (skip >= 1e11) break;
            t += skip;
            continue;
        }

        // Fine steps near the funnel, medium steps anywhere in the storm cell (so the wall cloud is
        // sampled consistently on both sides of the funnel bound), the regular cloud step elsewhere.
        float ds = cloudDs;
        if (cellStrength > 0.0 && cellFalloff((p - planetOffset).xz, WALL_R * 1.6) > 0.02) {
            ds = min(ds, max(60.0, t * 0.006));
        }
        if (nearFunnel) ds = min(ds, max(TORNADO_STEP, t * 0.003));
        vec3 sp = ro + rd * (t + ds * jitter);
        float sigmaC = inLayer ? cloudDensity(sp, true) : 0.0;
        float fS = 0.0;
        float dS = 0.0;
        if (nearFunnel) tornadoDensity(sp - planetOffset, baseY, topY, fS, dS);
        float sigma = sigmaC + fS + dS;

        if (sigma > 0.0) {
            if (abs(t - sunT) > 300.0) {
                vec2 sod = sunOpticalDepth(sp);
                sunColor = sod.x < 0.0 ? vec3(0.0) : extinction(sod);
                sunT = t;
            }
            // One light march through the combined medium: clouds shadow the funnel and vice versa.
            float lightOd = 0.0;
            float lt = 0.0;
            float lstep = CLOUD_LIGHT_STEP;
            for (int j = 0; j < CLOUD_LIGHT_STEPS; j++) {
                vec3 q = sp + sunDir * (lt + 0.5 * lstep);
                float od = cloudDensity(q, false);
                if (tornado) {
                    float lf;
                    float ld;
                    tornadoDensity(q - planetOffset, baseY, topY, lf, ld);
                    od += lf + ld;
                }
                lightOd += od * lstep;
                lt += lstep;
                lstep *= 2.0;
            }
            // Below the normal cloud base (funnel, debris, lowered wall cloud) the light march can't reach
            // the deck, so add its shadow explicitly. It varies slowly along the ray, so reuse it for ~150 m.
            float alt = length(sp) - R_PLANET;
            float deck = 1.0;
            if (alt < cloudBottom) {
                if (abs(t - deckT) > 150.0) {
                    deckCached = cloudShadow(sp);
                    deckT = t;
                }
                deck = deckCached;
            }

            // Octave 0 is single scattering; later octaves approximate light diffused inside the medium:
            // it penetrates further (lower extinction) and spreads out (less forward-peaked phase).
            float ms = 0.0;
            float ea = 1.0;
            float eb = 1.0;
            for (int n = 0; n < MS_OCTAVES; n++) {
                ms += eb * phases[n] * exp(-ea * lightOd);
                ea *= MS_A;
                eb *= MS_B;
            }
            // Debris is dust: broad HG lobe, single scattering, absorbing albedo.
            float share = dS / sigma;
            float sunScatter = mix(ms, dustPhase * exp(-lightOd), share);
            vec3 albedo = mix(vec3(1.0), DEBRIS_ALBEDO, share);

            float hFrac = clamp((alt - cloudBottom) / (cloudTop - cloudBottom), 0.0, 1.0);
            vec3 s = albedo * sunColor * (CLOUD_SUN * sunScatter * deck + ambient * mix(0.4, 1.0, hFrac));
            // Lightning diffuses through the medium from the flash point.
            s += albedo * FLASH_COLOR * flashIntensity * FLASH_CLOUD * exp(-length(sp - fp) / FLASH_RADIUS);

            float stepT = exp(-sigma * ds);
            float absorbed = trans * (1.0 - stepT);
            light += s * absorbed;
            depth += t * absorbed;
            weight += absorbed;
            trans *= stepT;
            if (trans < 0.01) break;
        }
        t += ds;
    }
    depth = weight > 0.0 ? depth / weight : start;
}

// Glow from the lightning bolt along this view ray, occluded beyond maxDist. Returns the radiance and
// the distance to the brightest segment.
vec3 boltLight(vec3 rd, float maxDist, out float boltDist) {
    boltDist = 1e12;
    if (boltSegCount == 0 || flashIntensity <= 0.0) return vec3(0.0);
    vec3 o = camPos * WORLD_SCALE; // local meters; avoids the planet-sized offset for precision
    float pixel = 2.0 * tanHalfFov / float(textureSize(texture0, 0).y);
    float best = 0.0;
    for (int i = 0; i < boltSegCount; i++) {
        vec3 a = boltSegs[2 * i] * WORLD_SCALE;
        vec3 ab = boltSegs[2 * i + 1] * WORLD_SCALE - a;
        vec3 w0 = o - a;
        float b = dot(rd, ab);
        float c = dot(ab, ab);
        float denom = c - b * b;
        float s = denom > 1e-6 ? clamp((dot(ab, w0) - b * dot(rd, w0)) / denom, 0.0, 1.0) : 0.0;
        vec3 q = a + ab * s;
        float t = dot(q - o, rd);
        if (t <= 0.0 || t > maxDist) continue;
        float d = length(o + rd * t - q);
        float px = t * pixel;
        float width = max(BOLT_WIDTH, px * 0.75);
        float v = exp(-(d * d) / (width * width)) + 0.12 * exp(-d / (px * 6.0 + 40.0));
        if (v > best) {
            best = v;
            boltDist = t;
        }
    }
    return FLASH_COLOR * BOLT_BRIGHTNESS * flashIntensity * best;
}

void main() {
    vec2 ndc = fragTexCoord * 2.0 - 1.0;
    vec3 rd = normalize(camForward + ndc.x * aspect * tanHalfFov * camRight + ndc.y * tanHalfFov * camUp);
    vec3 ro = camPos * WORLD_SCALE + vec3(0.0, R_PLANET, 0.0);

    vec3 scene = texture(texture0, fragTexCoord).rgb;
    float depth = texture(depthTex, fragTexCoord).r;

    vec3 inscatter;
    vec3 transmittance;
    vec3 hdr;
    float sceneDist = 1e12;

    if (depth >= 1.0) {
        float tMax = raySphere(ro, rd, R_ATMOS).y;
        float ground = raySphere(ro, rd, R_PLANET).x;
        if (ground > 0.0) tMax = ground;
        scatter(ro, rd, tMax, inscatter, transmittance);

        float sunDisc = smoothstep(0.9995, 0.9998, dot(rd, sunDir));
        hdr = inscatter + sunDisc * transmittance * SUN_INTENSITY;
    } else {
        float z = depth * 2.0 - 1.0;
        float linear = 2.0 * NEAR * FAR / (FAR + NEAR - z * (FAR - NEAR));
        float dist = linear / dot(rd, camForward) * WORLD_SCALE;
        sceneDist = dist;
        scatter(ro, rd, dist, inscatter, transmittance);

        vec3 surface = ro + rd * dist;
        vec2 sod = sunOpticalDepth(surface);
        vec3 sunLight = sod.x < 0.0 ? vec3(0.0) : extinction(sod) * cloudShadow(surface);
        vec3 flash = FLASH_COLOR * flashIntensity * FLASH_GROUND * exp(-length(surface - flashP()) / FLASH_GROUND_RADIUS);
        vec3 light = sunLight * SURFACE_LIGHT + skyAmbient() * 0.35 + flash;
        hdr = scene * light * transmittance + inscatter;
    }

    float boltDist;
    vec3 bolt = boltLight(rd, sceneDist, boltDist);

    vec3 cloudLight;
    float cloudTrans;
    float cloudDepth;
    marchClouds(ro, rd, sceneDist, cloudLight, cloudTrans, cloudDepth);

    if (cloudTrans < 1.0) {
        // Haze between the camera and the cloud.
        vec3 cloudInscatter;
        vec3 cloudAtmosT;
        scatter(ro, rd, cloudDepth, cloudInscatter, cloudAtmosT);
        hdr = cloudTrans * hdr + (1.0 - cloudTrans) * cloudInscatter + cloudAtmosT * cloudLight;
    }

    // Bolts in front of the cloud stay bright; behind it they glow through.
    hdr += boltDist < cloudDepth || cloudTrans >= 1.0 ? bolt : bolt * cloudTrans;


    finalColor = vec4(1.0 - exp(-hdr), 1.0);
}
