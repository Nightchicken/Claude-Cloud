const rl = @import("raylib");
const std = @import("std");
const cloudfield = @import("cloudfield.zig");
const weather = @import("weather.zig");

const cloud_data = struct {
    particle_count: u16,
    droplet_size: u16,
    water_content: f16,
    scatter_point: f16,
};

const Simplex = struct {
    perm: [512]u8,

    const F2: f32 = 0.5 * (@sqrt(3.0) - 1.0);
    const G2: f32 = (3.0 - @sqrt(3.0)) / 6.0;

    const grad2 = [8][2]f32{
        .{ 1, 1 }, .{ -1, 1 }, .{ 1, -1 }, .{ -1, -1 },
        .{ 1, 0 }, .{ -1, 0 }, .{ 0, 1 },  .{ 0, -1 },
    };

    pub fn init(seed: u64) Simplex {
        var p: [256]u8 = undefined;
        for (&p, 0..) |*v, i| v.* = @intCast(i);
        var prng = std.Random.DefaultPrng.init(seed);
        prng.random().shuffle(u8, &p);

        var s: Simplex = undefined;
        for (0..512) |i| s.perm[i] = p[i & 255];
        return s;
    }

    fn corner(self: *const Simplex, gi: usize, x: f32, y: f32) f32 {
        const t = 0.5 - x * x - y * y;
        if (t < 0) return 0;
        const g = grad2[self.perm[gi] & 7];
        const t2 = t * t;
        return t2 * t2 * (g[0] * x + g[1] * y);
    }

    /// 2D simplex noise in roughly [-1, 1].
    pub fn noise2(self: *const Simplex, x: f32, y: f32) f32 {
        const s = (x + y) * F2;
        const i = @floor(x + s);
        const j = @floor(y + s);
        const t = (i + j) * G2;
        const x0 = x - (i - t);
        const y0 = y - (j - t);

        const di: f32, const dj: f32 = if (x0 > y0) .{ 1, 0 } else .{ 0, 1 };

        const x1 = x0 - di + G2;
        const y1 = y0 - dj + G2;
        const x2 = x0 - 1.0 + 2.0 * G2;
        const y2 = y0 - 1.0 + 2.0 * G2;

        const ii: usize = @intCast(@as(i32, @intFromFloat(i)) & 255);
        const jj: usize = @intCast(@as(i32, @intFromFloat(j)) & 255);
        const di_i: usize = @intFromFloat(di);
        const dj_i: usize = @intFromFloat(dj);

        const n0 = self.corner(ii + self.perm[jj], x0, y0);
        const n1 = self.corner(ii + di_i + self.perm[jj + dj_i], x1, y1);
        const n2 = self.corner(ii + 1 + self.perm[jj + 1], x2, y2);

        return 70.0 * (n0 + n1 + n2);
    }

    /// Fractal Brownian motion: summed octaves of noise2, normalized to roughly [-1, 1].
    pub fn fbm(self: *const Simplex, x: f32, y: f32, octaves: u32) f32 {
        var sum: f32 = 0;
        var amp: f32 = 1;
        var freq: f32 = 1;
        var norm: f32 = 0;
        for (0..octaves) |_| {
            sum += amp * self.noise2(x * freq, y * freq);
            norm += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        return sum / norm;
    }
};

const map_size: usize = 256;
const water_level: f32 = 0.30;

fn heightColor(h: f32) rl.Color {
    if (h < water_level) return .{ .r = 40, .g = 80, .b = 160, .a = 255 };
    if (h < 0.35) return .{ .r = 200, .g = 190, .b = 140, .a = 255 };
    if (h < 0.60) return .{ .r = 70, .g = 140, .b = 60, .a = 255 };
    if (h < 0.78) return .{ .r = 120, .g = 110, .b = 100, .a = 255 };
    return .{ .r = 240, .g = 240, .b = 245, .a = 255 };
}

fn shade(c: rl.Color, k: f32) rl.Color {
    return .{
        .r = @intFromFloat(@as(f32, @floatFromInt(c.r)) * k),
        .g = @intFromFloat(@as(f32, @floatFromInt(c.g)) * k),
        .b = @intFromFloat(@as(f32, @floatFromInt(c.b)) * k),
        .a = 255,
    };
}

/// Render target with a sampleable depth texture (raylib's default uses a depth renderbuffer).
fn loadSceneTarget(w: i32, h: i32) !rl.RenderTexture2D {
    const gl = rl.gl;
    const id = gl.rlLoadFramebuffer();
    if (id == 0) return error.FramebufferCreateFailed;
    gl.rlEnableFramebuffer(id);
    defer gl.rlDisableFramebuffer();

    const target = rl.RenderTexture2D{
        .id = id,
        .texture = .{
            .id = gl.rlLoadTexture(null, w, h, @intFromEnum(rl.PixelFormat.uncompressed_r8g8b8a8), 1),
            .width = w,
            .height = h,
            .mipmaps = 1,
            .format = .uncompressed_r8g8b8a8,
        },
        .depth = .{
            .id = gl.rlLoadTextureDepth(w, h, false),
            .width = w,
            .height = h,
            .mipmaps = 1,
            .format = .uncompressed_r32,
        },
    };
    const tex2d = @intFromEnum(gl.rlFramebufferAttachTextureType.rl_attachment_texture2d);
    gl.rlFramebufferAttach(id, target.texture.id, @intFromEnum(gl.rlFramebufferAttachType.rl_attachment_color_channel0), tex2d, 0);
    gl.rlFramebufferAttach(id, target.depth.id, @intFromEnum(gl.rlFramebufferAttachType.rl_attachment_depth), tex2d, 0);
    if (!gl.rlFramebufferComplete(id)) return error.FramebufferIncomplete;
    return target;
}

const AtmosphereShader = struct {
    shader: rl.Shader,
    depth_tex: i32,
    cam_pos: i32,
    cam_forward: i32,
    cam_right: i32,
    cam_up: i32,
    tan_half_fov: i32,
    aspect: i32,
    sun_dir: i32,
    time: i32,
    cloud_coverage: i32,
    cloud_density_scale: i32,
    droplet_diameter: i32,
    cloud_bottom: i32,
    cloud_top: i32,
    cloud_scale: i32,
    cloud_shape_top: i32,
    cloud_offset: i32,
    wind_dir: i32,
    flash_pos: i32,
    flash_intensity: i32,
    bolt_segs: i32,
    bolt_seg_count: i32,
    tornado_pos: i32,
    tornado_lean: i32,
    tornado_strength: i32,
    tornado_touchdown: i32,
    cell_pos: i32,
    cell_strength: i32,
    cell_age: i32,

    fn load() !AtmosphereShader {
        const shader = try rl.loadShaderFromMemory(null, @embedFile("atmosphere.fs"));
        return .{
            .shader = shader,
            .depth_tex = rl.getShaderLocation(shader, "depthTex"),
            .cam_pos = rl.getShaderLocation(shader, "camPos"),
            .cam_forward = rl.getShaderLocation(shader, "camForward"),
            .cam_right = rl.getShaderLocation(shader, "camRight"),
            .cam_up = rl.getShaderLocation(shader, "camUp"),
            .tan_half_fov = rl.getShaderLocation(shader, "tanHalfFov"),
            .aspect = rl.getShaderLocation(shader, "aspect"),
            .sun_dir = rl.getShaderLocation(shader, "sunDir"),
            .time = rl.getShaderLocation(shader, "time"),
            .cloud_coverage = rl.getShaderLocation(shader, "cloudCoverage"),
            .cloud_density_scale = rl.getShaderLocation(shader, "cloudDensityScale"),
            .droplet_diameter = rl.getShaderLocation(shader, "dropletDiameter"),
            .cloud_bottom = rl.getShaderLocation(shader, "cloudBottom"),
            .cloud_top = rl.getShaderLocation(shader, "cloudTop"),
            .cloud_scale = rl.getShaderLocation(shader, "cloudScale"),
            .cloud_shape_top = rl.getShaderLocation(shader, "cloudShapeTop"),
            .cloud_offset = rl.getShaderLocation(shader, "cloudOffset"),
            .wind_dir = rl.getShaderLocation(shader, "windDir"),
            .flash_pos = rl.getShaderLocation(shader, "flashPos"),
            .flash_intensity = rl.getShaderLocation(shader, "flashIntensity"),
            .bolt_segs = rl.getShaderLocation(shader, "boltSegs"),
            .bolt_seg_count = rl.getShaderLocation(shader, "boltSegCount"),
            .tornado_pos = rl.getShaderLocation(shader, "tornadoPos"),
            .tornado_lean = rl.getShaderLocation(shader, "tornadoLean"),
            .tornado_strength = rl.getShaderLocation(shader, "tornadoStrength"),
            .tornado_touchdown = rl.getShaderLocation(shader, "tornadoTouchdown"),
            .cell_pos = rl.getShaderLocation(shader, "cellPos"),
            .cell_strength = rl.getShaderLocation(shader, "cellStrength"),
            .cell_age = rl.getShaderLocation(shader, "cellAge"),
        };
    }

    fn setCamera(self: AtmosphereShader, camera: rl.Camera3D, aspect: f32) void {
        const forward = camera.target.subtract(camera.position).normalize();
        const right = forward.crossProduct(camera.up).normalize();
        const up = right.crossProduct(forward);
        const tan_half_fov = @tan(camera.fovy * std.math.rad_per_deg * 0.5);
        rl.setShaderValue(self.shader, self.cam_pos, &camera.position, .vec3);
        rl.setShaderValue(self.shader, self.cam_forward, &forward, .vec3);
        rl.setShaderValue(self.shader, self.cam_right, &right, .vec3);
        rl.setShaderValue(self.shader, self.cam_up, &up, .vec3);
        rl.setShaderValue(self.shader, self.tan_half_fov, &tan_half_fov, .float);
        rl.setShaderValue(self.shader, self.aspect, &aspect, .float);
    }

    fn setSun(self: AtmosphereShader, dir: rl.Vector3) void {
        rl.setShaderValue(self.shader, self.sun_dir, &dir, .vec3);
    }

    fn setTime(self: AtmosphereShader, t: f32) void {
        rl.setShaderValue(self.shader, self.time, &t, .float);
    }

    fn setWeather(self: AtmosphereShader, params: cloudfield.Params, droplet_diameter: f32) void {
        rl.setShaderValue(self.shader, self.cloud_coverage, &params.coverage, .float);
        rl.setShaderValue(self.shader, self.cloud_density_scale, &params.density_scale, .float);
        rl.setShaderValue(self.shader, self.droplet_diameter, &droplet_diameter, .float);
        rl.setShaderValue(self.shader, self.cloud_bottom, &params.bottom, .float);
        rl.setShaderValue(self.shader, self.cloud_top, &params.top, .float);
        rl.setShaderValue(self.shader, self.cloud_scale, &params.scale, .float);
        rl.setShaderValue(self.shader, self.cloud_shape_top, &params.shape_top, .float);
        rl.setShaderValue(self.shader, self.cloud_offset, &params.offset, .vec2);
        rl.setShaderValue(self.shader, self.wind_dir, &params.wind_dir, .vec2);
    }

    fn setTornado(self: AtmosphereShader, tornado: weather.Tornado, ground: weather.Ground) void {
        const pos = rl.Vector3.init(tornado.pos[0], ground.heightAt(tornado.pos[0], tornado.pos[1]), tornado.pos[1]);
        const lean = tornado.lean();
        const strength = tornado.strength();
        const touchdown = tornado.touchdown();
        rl.setShaderValue(self.shader, self.tornado_pos, &pos, .vec3);
        rl.setShaderValue(self.shader, self.tornado_lean, &lean, .vec2);
        rl.setShaderValue(self.shader, self.tornado_strength, &strength, .float);
        rl.setShaderValue(self.shader, self.tornado_touchdown, &touchdown, .float);
        const cell_pos = tornado.cellPos();
        const cell_strength = tornado.cellStrength();
        rl.setShaderValue(self.shader, self.cell_pos, &cell_pos, .vec2);
        rl.setShaderValue(self.shader, self.cell_strength, &cell_strength, .float);
        rl.setShaderValue(self.shader, self.cell_age, &tornado.age, .float);
    }

    fn setLightning(self: AtmosphereShader, lightning: weather.Lightning) void {
        const intensity = lightning.intensity();
        const segs = lightning.visibleSegments();
        const count: i32 = @intCast(segs.len / 2);
        rl.setShaderValue(self.shader, self.flash_pos, &lightning.flash_pos, .vec3);
        rl.setShaderValue(self.shader, self.flash_intensity, &intensity, .float);
        rl.setShaderValue(self.shader, self.bolt_seg_count, &count, .int);
        if (segs.len > 0) rl.setShaderValueV(self.shader, self.bolt_segs, segs.ptr, .vec3, @intCast(segs.len));
    }
};

/// Everything needed to draw one frame of the scene, so the window and wallpaper capture share it.
const View = struct {
    atmosphere: AtmosphereShader,
    terrain: rl.Model,
    camera: rl.Camera3D,
    sun_elevation: f32,
    time: f32,
    params: cloudfield.Params,
    droplet_diameter: f32,
    lightning: *const weather.Lightning,
    tornado: *const weather.Tornado,
    ground: weather.Ground,

    /// Render the terrain's color and depth into `target`.
    fn renderTerrain(self: View, target: rl.RenderTexture2D) void {
        rl.beginTextureMode(target);
        rl.clearBackground(.black);
        rl.beginMode3D(self.camera);
        rl.drawModel(self.terrain, .{ .x = 0, .y = 0, .z = 0 }, 1, .white);
        rl.endMode3D();
        rl.endTextureMode();
    }

    /// Full-screen atmosphere/cloud pass reading `target`, drawn at w x h into whatever is bound.
    fn drawAtmosphere(self: View, target: rl.RenderTexture2D, w: i32, h: i32) void {
        const a = self.atmosphere;
        rl.beginShaderMode(a.shader);
        a.setCamera(self.camera, @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h)));
        a.setSun(sunDirection(self.sun_elevation));
        a.setTime(self.time);
        a.setWeather(self.params, self.droplet_diameter);
        a.setLightning(self.lightning.*);
        a.setTornado(self.tornado.*, self.ground);
        rl.setShaderValueTexture(a.shader, a.depth_tex, target.depth);
        // Negative height flips the render texture, which is stored upside down.
        rl.drawTextureRec(
            target.texture,
            .{ .x = 0, .y = 0, .width = @floatFromInt(w), .height = @floatFromInt(-h) },
            .{ .x = 0, .y = 0 },
            .white,
        );
        rl.endShaderMode();
    }
};

/// Frames averaged per wallpaper; each has a different ray-march jitter, so averaging removes the grain.
const wallpaper_frames = 24;

/// First unused ~/Pictures/clouds_NNN.png.
fn nextWallpaperPath(buf: []u8) ![:0]const u8 {
    const home = std.c.getenv("HOME") orelse return error.NoHomeDirectory;
    var i: u32 = 1;
    while (i < 1000) : (i += 1) {
        const path = try std.fmt.bufPrintZ(buf, "{s}/Pictures/clouds_{d:0>3}.png", .{ std.mem.span(home), i });
        if (!rl.fileExists(path)) return path;
    }
    return error.TooManyWallpapers;
}

/// Render `view` at the monitor's resolution, without the HUD, averaging several frames, and save a PNG.
fn saveWallpaper(view: View, path: [:0]const u8) !void {
    const monitor = rl.getCurrentMonitor();
    var w = rl.getMonitorWidth(monitor);
    var h = rl.getMonitorHeight(monitor);
    if (w <= 0 or h <= 0) {
        w = 2560;
        h = 1600;
    }
    const target = try loadSceneTarget(w, h);
    defer rl.unloadRenderTexture(target);
    const out = try rl.loadRenderTexture(w, h);
    defer rl.unloadRenderTexture(out);

    const n: usize = @intCast(w * h);
    const accum = try std.heap.page_allocator.alloc(f32, n * 3);
    defer std.heap.page_allocator.free(accum);
    @memset(accum, 0);

    var last: ?rl.Image = null;
    defer if (last) |img| rl.unloadImage(img);
    for (0..wallpaper_frames) |i| {
        var v = view;
        v.time = view.time + @as(f32, @floatFromInt(i)) / 60.0;
        v.renderTerrain(target);
        rl.beginTextureMode(out);
        v.drawAtmosphere(target, w, h);
        rl.endTextureMode();

        if (last) |img| rl.unloadImage(img);
        last = try rl.loadImageFromTexture(out.texture);
        const px: [*]u8 = @ptrCast(last.?.data);
        for (0..n) |k| {
            accum[k * 3] += @floatFromInt(px[k * 4]);
            accum[k * 3 + 1] += @floatFromInt(px[k * 4 + 1]);
            accum[k * 3 + 2] += @floatFromInt(px[k * 4 + 2]);
        }
    }

    var img = last.?;
    const px: [*]u8 = @ptrCast(img.data);
    for (0..n) |k| {
        px[k * 4] = @intFromFloat(accum[k * 3] / wallpaper_frames);
        px[k * 4 + 1] = @intFromFloat(accum[k * 3 + 1] / wallpaper_frames);
        px[k * 4 + 2] = @intFromFloat(accum[k * 3 + 2] / wallpaper_frames);
        px[k * 4 + 3] = 255;
    }
    rl.imageFlipVertical(&img);
    last = img;
    if (!rl.exportImage(img, path)) return error.ExportFailed;
}

/// Scripted showcase: a held supercell with a tornado kept centered by an orbiting camera,
/// lightning, and the sun sweeping from afternoon down to sunset and back.
fn runDemo(
    t: f32,
    camera: *rl.Camera3D,
    sky: *weather.Weather,
    tornado: *weather.Tornado,
    lightning: *weather.Lightning,
    sun_elevation: *f32,
    params: cloudfield.Params,
    ground: weather.Ground,
) void {
    sky.water_target = 0.88;
    sky.type_target = 1.0;
    sky.water += (sky.water_target - sky.water) * 0.05;
    sky.cloud_type += (sky.type_target - sky.cloud_type) * 0.05;

    if (!tornado.active) {
        tornado.spawnAt(15, 15);
        tornado.lifetime = 150;
    }
    if (lightning.age > 7) lightning.strikeAtWettest(params, ground, true);

    sun_elevation.* = 14 + 12 * @cos(t * 0.04);

    // Orbit the tornado, aimed at the middle of the funnel so it stays centered.
    const base_y = ground.heightAt(tornado.pos[0], tornado.pos[1]);
    const mid_y = (base_y + params.bottom / weather.world_scale) * 0.5;
    const angle = 0.6 + t * 0.06;
    const radius: f32 = 34;
    camera.position = .{
        .x = tornado.pos[0] + @cos(angle) * radius,
        .y = mid_y + 3,
        .z = tornado.pos[1] + @sin(angle) * radius,
    };
    camera.target = .{ .x = tornado.pos[0], .y = mid_y, .z = tornado.pos[1] };
}

/// Sun direction from elevation (degrees) along the baked-lighting azimuth.
fn sunDirection(elevation_deg: f32) rl.Vector3 {
    const el = elevation_deg * std.math.rad_per_deg;
    const az = rl.Vector2.init(-0.5, -0.3).normalize();
    return rl.Vector3.init(@cos(el) * az.x, @sin(el), @cos(el) * az.y);
}

pub fn main(init: std.process.Init.Minimal) !void {
    // `--demo` starts in demo mode (also toggled with M).
    var demo = false;
    var args = init.args.iterate();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--demo")) demo = true;
    }

    const width = 1280;
    const height = 720;
    const terrain_size: f32 = 200;
    const terrain_height: f32 = 30;
    const noise_scale: f32 = 4.0 / @as(f32, map_size);
    const mouse_sensitivity: f32 = 0.05;
    // Movement, in world units (100 m) per second.
    const fly_speed_per_height: f32 = 0.8;
    const fly_speed_min: f32 = 0.05;
    const fly_speed_max: f32 = 40;
    const eye_height: f32 = 0.1; // 10 m; keeps the ground beyond raylib's 5 m near clip plane
    const max_altitude: f32 = 150; // 15 km

    rl.initWindow(width, height, "test");
    rl.setTargetFPS(60);
    defer rl.closeWindow();

    // Clouds sit at 35-50 world units (3.5-5 km); Tab swaps between a view above and below them.
    const above_view = [2]rl.Vector3{ .{ .x = 0, .y = 70, .z = 130 }, .{ .x = 0, .y = 0, .z = 0 } };
    // Below view faces the sun's azimuth so light shafts are in frame.
    const below_view = [2]rl.Vector3{ .{ .x = 0, .y = 22, .z = 95 }, .{ .x = -43, .y = 34, .z = 70 } };
    var below = false;
    var camera = rl.Camera3D{
        .position = above_view[0],
        .target = above_view[1],
        .up = .{ .x = 0, .y = 1, .z = 0 },
        .fovy = 60,
        .projection = .perspective,
    };
    rl.disableCursor();

    const noise = Simplex.init(1337);

    var heights: [map_size * map_size]f32 = undefined;
    for (0..map_size) |z| {
        for (0..map_size) |x| {
            const n = noise.fbm(@as(f32, @floatFromInt(x)) * noise_scale, @as(f32, @floatFromInt(z)) * noise_scale, 6);
            heights[z * map_size + x] = std.math.clamp((n + 1.0) * 0.5, 0.0, 1.0);
        }
    }

    var color_img = rl.genImageColor(map_size, map_size, .black);
    defer rl.unloadImage(color_img);

    // Bake simple directional lighting into the color texture since the default shader is unlit.
    const light = rl.Vector3.init(-0.5, 1.0, -0.3).normalize();
    const cell = terrain_size / @as(f32, map_size - 1);
    for (0..map_size) |z| {
        for (0..map_size) |x| {
            const h = heights[z * map_size + x];
            const xl = heights[z * map_size + (x -| 1)];
            const xr = heights[z * map_size + @min(x + 1, map_size - 1)];
            const zu = heights[(z -| 1) * map_size + x];
            const zd = heights[@min(z + 1, map_size - 1) * map_size + x];
            const normal = rl.Vector3.init(
                (xl - xr) * terrain_height / (2 * cell),
                1.0,
                (zu - zd) * terrain_height / (2 * cell),
            ).normalize();
            const k = 0.35 + 0.65 * @max(normal.dotProduct(light), 0.0);

            rl.imageDrawPixel(&color_img, @intCast(x), @intCast(z), shade(heightColor(h), k));
        }
    }

    var texture = try rl.loadTextureFromImage(color_img);
    defer rl.unloadTexture(texture);
    rl.genTextureMipmaps(&texture);
    rl.setTextureFilter(texture, .trilinear);

    // genMeshHeightmap receives a corrupted `size` under Zig 0.16's C ABI, so displace a plane instead.
    // The plane's vertices are laid out as x + z * map_size, matching `heights`.
    const mesh = rl.genMeshPlane(terrain_size, terrain_size, map_size - 1, map_size - 1);
    for (heights, 0..) |h, i| mesh.vertices[i * 3 + 1] = @max(h, water_level) * terrain_height;
    rl.updateMeshBuffer(mesh, 0, mesh.vertices, @intCast(map_size * map_size * 3 * @sizeOf(f32)), 0);

    const terrain = try rl.loadModelFromMesh(mesh);
    defer rl.unloadModel(terrain);
    terrain.materials[0].maps[@intFromEnum(rl.MaterialMapIndex.albedo)].texture = texture;

    const atmosphere = try AtmosphereShader.load();
    defer rl.unloadShader(atmosphere.shader);
    const scene = try loadSceneTarget(width, height);
    defer rl.unloadRenderTexture(scene);
    var sun_elevation: f32 = 20;

    // Water content (0..1) drives cloud thickness and lightning; droplet size drives scattering.
    // Weather evolves water content, cloud type and wind over time.
    var clouds = cloud_data{ .particle_count = 0, .droplet_size = 20, .water_content = 0.4, .scatter_point = 0 };
    var sky = weather.Weather.init(7);
    var lightning = weather.Lightning.init(42);
    var tornado = weather.Tornado.init(99);
    const ground = weather.Ground{
        .heights = &heights,
        .map_size = map_size,
        .size = terrain_size,
        .height = terrain_height,
        .water_level = water_level,
    };

    var demo_time: f32 = 0;
    var released = false;
    var skip_mouse_frames: u32 = 0;
    var saved_buf: [512]u8 = undefined;
    var saved_msg: [:0]const u8 = "";
    var saved_timer: f32 = 0;

    while (!rl.windowShouldClose()) {
        // J releases the mouse and ignores all controls (to screenshot or move the window);
        // clicking back inside the window recaptures them.
        if (!released and rl.isKeyPressed(.j)) {
            released = true;
            rl.enableCursor();
        } else if (released and rl.isWindowFocused() and rl.isMouseButtonPressed(.left)) {
            released = false;
            rl.disableCursor();
            skip_mouse_frames = 2; // recapturing warps the cursor; don't turn that into a camera jump
        }
        const controls = !released;

        if (controls and rl.isKeyPressed(.m)) demo = !demo;
        if (!demo and controls) {
            var md = rl.getMouseDelta();
            if (skip_mouse_frames > 0) {
                skip_mouse_frames -= 1;
                md = rl.Vector2.init(0, 0);
            }
            rl.updateCameraPro(
                &camera,
                .{ .x = 0, .y = 0, .z = 0 },
                .{ .x = md.x * mouse_sensitivity, .y = md.y * mouse_sensitivity, .z = 0 },
                0,
            );
        }

        const dt = @min(rl.getFrameTime(), 0.1); // a wallpaper save stalls one frame; don't jump the sim
        if (controls and rl.isKeyDown(.up)) sun_elevation += 20 * dt;
        if (controls and rl.isKeyDown(.down)) sun_elevation -= 20 * dt;
        sun_elevation = std.math.clamp(sun_elevation, -10, 90);
        // F fast-forwards the weather (not the lightning flicker).
        const weather_dt = if (controls and rl.isKeyDown(.f)) dt * 15 else dt;
        sky.update(weather_dt);
        if (controls and rl.isKeyDown(.right_bracket)) sky.nudgeWater(0.25 * dt);
        if (controls and rl.isKeyDown(.left_bracket)) sky.nudgeWater(-0.25 * dt);
        clouds.water_content = @floatCast(sky.water);
        const water: f32 = clouds.water_content;

        const t: f32 = @floatCast(rl.getTime());
        var params = sky.params();
        tornado.update(weather_dt, sky, params);
        if (controls and rl.isKeyPressed(.t)) {
            // Spawn 5 km ahead of the camera, on the ground.
            const fwd = rl.Vector2.init(camera.target.x - camera.position.x, camera.target.z - camera.position.z).normalize();
            tornado.spawnAt(camera.position.x + fwd.x * 50, camera.position.z + fwd.y * 50);
        }
        // The tornado's parent cell boosts cloud water around it; lightning follows.
        params.cell_pos = tornado.cellPos();
        params.cell_strength = tornado.cellStrength();

        lightning.update(dt, water, params, ground);
        if (controls and rl.isKeyPressed(.l)) lightning.strikeAtWettest(params, ground, true);

        if (demo) {
            demo_time += dt;
            runDemo(demo_time, &camera, &sky, &tornado, &lightning, &sun_elevation, params, ground);
        }

        // Free-fly: WASD along the view, Space/Ctrl up/down, Shift boost. Speed scales with height
        // above ground so it's quick at altitude and walking-pace near the surface.
        {
            const forward = camera.target.subtract(camera.position).normalize();
            const right = forward.crossProduct(camera.up).normalize();
            var move = rl.Vector3.init(0, 0, 0);
            if (controls and rl.isKeyDown(.w)) move = move.add(forward);
            if (controls and rl.isKeyDown(.s)) move = move.subtract(forward);
            if (controls and rl.isKeyDown(.d)) move = move.add(right);
            if (controls and rl.isKeyDown(.a)) move = move.subtract(right);
            if (controls and rl.isKeyDown(.space)) move.y += 1;
            if (controls and rl.isKeyDown(.left_control)) move.y -= 1;
            if (move.length() > 0) {
                const clearance = camera.position.y - ground.heightAt(camera.position.x, camera.position.z);
                var speed = std.math.clamp(clearance * fly_speed_per_height, fly_speed_min, fly_speed_max);
                if (controls and rl.isKeyDown(.left_shift)) speed *= 4;
                const delta = move.normalize().scale(speed * dt);
                camera.position = camera.position.add(delta);
                camera.target = camera.target.add(delta);
            }

            const min_y = ground.heightAt(camera.position.x, camera.position.z) + eye_height;
            const max_y = max_altitude;
            const y = std.math.clamp(camera.position.y, min_y, max_y);
            camera.target.y += y - camera.position.y;
            camera.position.y = y;
        }

        if (controls and rl.isKeyPressed(.tab)) {
            below = !below;
            const view = if (below) below_view else above_view;
            camera.position = view[0];
            camera.target = view[1];
        }

        const view = View{
            .atmosphere = atmosphere,
            .terrain = terrain,
            .camera = camera,
            .sun_elevation = sun_elevation,
            .time = t,
            .params = params,
            .droplet_diameter = @floatFromInt(clouds.droplet_size),
            .lightning = &lightning,
            .tornado = &tornado,
            .ground = ground,
        };

        // P saves the current view as a full-resolution wallpaper (works while controls are released too).
        if (rl.isKeyPressed(.p)) {
            saved_timer = 5;
            if (nextWallpaperPath(saved_buf[0 .. saved_buf.len - 64])) |path| {
                if (saveWallpaper(view, path)) {
                    saved_msg = std.fmt.bufPrintZ(saved_buf[saved_buf.len - 64 ..], "Saved {s}", .{std.fs.path.basename(path)}) catch "Saved";
                } else |_| saved_msg = "Wallpaper save failed";
            } else |_| saved_msg = "Wallpaper save failed (no ~/Pictures path)";
        }
        saved_timer = @max(saved_timer - dt, 0);

        view.renderTerrain(scene);

        rl.beginDrawing();
        defer rl.endDrawing();

        view.drawAtmosphere(scene, width, height);

        rl.drawFPS(10, 10);
        if (demo) rl.drawText("DEMO  (M to exit)", width - 230, 10, 24, .yellow);
        if (released) rl.drawText("Controls released - click the window to resume", width / 2 - 270, height - 40, 24, .yellow);
        if (saved_timer > 0) rl.drawText(saved_msg, width / 2 - 200, height - 70, 24, .green);
        rl.drawText("WASD/Space/Ctrl: fly (Shift: fast)   Tab: preset views   M: demo   J: release mouse   P: save wallpaper   Up/Down: sun   [ ]: water   L: lightning   T: tornado   F: fast-forward weather", 10, 34, 20, .white);
        const state = if (water >= cloudfield.lightning_min_water) "storm" else "calm";
        rl.drawText(rl.textFormat("Water %.2f (%s)   %s   wind %.0f m/s", .{ water, state.ptr, sky.typeName().ptr, sky.wind_speed }), 10, 58, 20, .white);
        if (tornado.active) {
            const dx = tornado.pos[0] - camera.position.x;
            const dz = tornado.pos[1] - camera.position.z;
            const km = @sqrt(dx * dx + dz * dz) * weather.world_scale / 1000;
            rl.drawText(rl.textFormat("TORNADO %.1f km away", .{km}), 10, 82, 20, .red);
        }
    }
}
