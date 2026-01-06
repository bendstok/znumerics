const std = @import("std");

pub const PID_DEO_Sim = struct {
    K_p: f64,
    K_i: f64,
    K_d: f64,
    dt: f64,
    clamp_min: f64,
    clamp_max: f64,
    prev_e: f64,
    integral: f64,
    prev_y: f64 = 0.0,

    pub fn init(K_p: f64, T_i: f64, T_d: f64, dt: f64, clamp_min: f64, clamp_max: f64) PID_DEO_Sim {
        return PID_DEO_Sim{
            .K_p = K_p,
            .K_i = if (T_i != 0.0) K_p / T_i else 0.0,
            .K_d = K_p * T_d,
            .dt = dt,
            .clamp_min = clamp_min,
            .clamp_max = clamp_max,
            .prev_e = 0.0,
            .integral = 0.0,
        };
    }

    pub fn reset(self: *PID_DEO_Sim) void {
        self.prev_e = 0.0;
        self.integral = 0.0;
    }

    pub fn compute(self: *PID_DEO_Sim, ref: f64, y: f64) f64 {
        const err = ref - y;

        const P = self.K_p * err;

        const D = -self.K_d * ((y - self.prev_y) / self.dt);
        self.prev_y = y;

        const prev_integral = self.integral;
        self.integral += err * self.dt;
        var I: f64 = 0.0;

        var avoid_windup = false;
        if (P + I + D >= self.clamp_max or P + I + D <= self.clamp_min) {
            avoid_windup = true;
        }

        if (avoid_windup) {
            // If we hit a windup
            // we use the integral value before
            // we hit the windup
            self.integral = prev_integral;
            I = self.K_i * self.integral;
        } else {
            I = self.K_i * self.integral;
        }

        self.prev_e = err;

        var output = P + I + D;
        if (output < self.clamp_min) output = self.clamp_min;
        if (output > self.clamp_max) output = self.clamp_max;

        return output;
    }
};
