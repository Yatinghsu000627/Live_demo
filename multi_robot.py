import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.transforms as transforms
from collections import deque


class Entity:
    def __init__(self, name, pos, signal_range, gain, color='r'):
        self.name = name
        self.pos = np.array(pos, dtype=float)
        self.signal_range = signal_range
        self.gain = gain
        self.color = color

    def field(self, X, Y):
        sigma = self.signal_range / 2.0
        distance_sq = (X - self.pos[0])**2 + (Y - self.pos[1])**2
        return self.gain * np.exp(-distance_sq / (2 * sigma**2))


class Robot(Entity):
    def __init__(self, name, path, signal_range, gain, color='b'):
        super().__init__(name, path[0], signal_range, gain, color)
        self.path = path

    def update_position(self, i):
        if i < len(self.path):
            self.pos = np.array(self.path[i])


class Obstacle:
    def __init__(self, center, width, height, angle_deg, attenuation):
        self.center = np.array(center, dtype=float)
        self.width = width
        self.height = height
        self.angle_deg = angle_deg
        self.attenuation = attenuation

    def get_vertices(self):
        cx, cy = self.center
        w, h, angle_rad = self.width, self.height, np.deg2rad(self.angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        half_w, half_h = w / 2, h / 2
        local_vertices = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        world_vertices = [(x * cos_a - y * sin_a + cx, x * sin_a + y * cos_a + cy) for x, y in local_vertices]

        return world_vertices

    def create_inside_mask(self, X, Y):
        cx, cy = self.center
        angle_rad = np.deg2rad(self.angle_deg)
        x_t = X - cx
        y_t = Y - cy
        cos_t, sin_t = np.cos(-angle_rad), np.sin(-angle_rad)
        x_local = x_t * cos_t - y_t * sin_t
        y_local = x_t * sin_t + y_t * cos_t
        mask = (np.abs(x_local) < self.width / 2) & (np.abs(y_local) < self.height / 2)

        return mask

    def _is_point_inside(self, p):
        cx, cy = self.center
        angle_rad = np.deg2rad(self.angle_deg)
        p_t = p - self.center
        cos_t, sin_t = np.cos(-angle_rad), np.sin(-angle_rad)
        p_local_x = p_t[0] * cos_t - p_t[1] * sin_t
        p_local_y = p_t[0] * sin_t + p_t[1] * cos_t
        
        return (np.abs(p_local_x) < self.width / 2) and (np.abs(p_local_y) < self.height / 2)

    def line_attenuation(self, p1, p2, num_samples=10):
        samples_x = np.linspace(p1[0], p2[0], num_samples)
        samples_y = np.linspace(p1[1], p2[1], num_samples)
        
        for i in range(num_samples):
            p = np.array([samples_x[i], samples_y[i]])
            if self._is_point_inside(p):
                return self.attenuation
        
        return 1.0

    def create_field_attenuation_mask(self, X, Y, source_pos):
        s_x, s_y = source_pos
        
        vertices = self.get_vertices()
        vertex_angles = [np.arctan2(v[1] - s_y, v[0] - s_x) for v in vertices]

        angle_center = np.arctan2(self.center[1] - s_y, self.center[0] - s_x)
        relative_angles = (np.array(vertex_angles) - angle_center + np.pi) % (2 * np.pi) - np.pi
        
        min_rel_angle = np.min(relative_angles)
        max_rel_angle = np.max(relative_angles)
        
        shadow_angle_1 = (angle_center + min_rel_angle + np.pi) % (2 * np.pi) - np.pi
        shadow_angle_2 = (angle_center + max_rel_angle + np.pi) % (2 * np.pi) - np.pi

        Grid_Angles = np.arctan2(Y - s_y, X - s_x)
        
        a1, a2 = shadow_angle_1, shadow_angle_2
        if a1 <= a2:
            is_in_wedge = (Grid_Angles >= a1) & (Grid_Angles <= a2)
        else: 
            is_in_wedge = (Grid_Angles >= a1) | (Grid_Angles <= a2)

        vertex_dists_sq = [(v[0] - s_x)**2 + (v[1] - s_y)**2 for v in vertices]
        min_dist_sq = np.min(vertex_dists_sq)
        Grid_Dist_Sq = (Y - s_y)**2 + (X - s_x)**2
        is_behind = (Grid_Dist_Sq >= min_dist_sq * 0.95) 

        is_inside_obstacle = self.create_inside_mask(X, Y)

        shadow_mask_bool = (is_in_wedge & is_behind) | is_inside_obstacle
        
        mask = np.ones_like(X)
        mask[shadow_mask_bool] = self.attenuation
        
        return mask
    

class World:
    def __init__(self, xlim=(-12,12), ylim=(-12,12), resolution=100):
        self.x_grid = np.linspace(xlim[0], xlim[1], resolution)
        self.y_grid = np.linspace(ylim[0], ylim[1], resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        self.obstacles = []
        self.agents = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def add_agent(self, agent):
        self.agents.append(agent)

    def link_strength(self, a, b):
        dist = np.linalg.norm(a.pos - b.pos)
        sigma = a.signal_range / 2.0
        s = a.gain * np.exp(-dist**2 / (2 * sigma**2))
        
        final_attenuation = 1.0
        for obs in self.obstacles:
            final_attenuation *= obs.line_attenuation(a.pos, b.pos)
        
        return s * final_attenuation

    def connected_robots(self, threshold=0.02):
        if not self.agents:
            return set()

        n = len(self.agents)
        adj = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                s_ij = self.link_strength(self.agents[i], self.agents[j])
                s_ji = self.link_strength(self.agents[j], self.agents[i])
                if max(s_ij, s_ji) > threshold:
                    adj[i].add(j)
                    adj[j].add(i)

        visited = set([0])
        q = deque([0])
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        
        return visited


    def compute_signal_field(self, max_strength=1.0):
        connected = self.connected_robots()
        
        Z_total = np.zeros_like(self.X)
        
        for i, agent in enumerate(self.agents):
            if i == 0 or i in connected:
                Z = agent.field(self.X, self.Y)
                for obs in self.obstacles:
                    shadow_mask = obs.create_field_attenuation_mask(self.X, self.Y, agent.pos)
                    Z *= shadow_mask
                Z_total += Z
        
        return np.clip(Z_total, 0, max_strength)


class Visualizer:
    def __init__(self, world, fps=25):
        self.world = world
        self.fps = fps
        self.fig, self.ax = plt.subplots(figsize=(9,8))
        self.heatmap = None
        self.agent_plots = {}
        self.links = [] 

    def setup(self):
        ax = self.ax
        ax.set_xlim(-12,12); ax.set_ylim(-12,12)
        ax.set_aspect('equal')
        # ax.grid(True, linestyle=':')
        
        for obs in self.world.obstacles:
            rect = patches.Rectangle(
                (-obs.width/2, -obs.height/2),
                obs.width, obs.height,
                facecolor='gray', edgecolor='black', alpha=0.6, label='Obstacle'
            )
            transform = (
                transforms.Affine2D()
                .rotate_deg_around(0,0,obs.angle_deg)
                + transforms.Affine2D().translate(obs.center[0], obs.center[1])
                + ax.transData
            )
            rect.set_transform(transform)
            ax.add_patch(rect)
            
        Z_init = self.world.compute_signal_field()
        self.heatmap = ax.pcolormesh(
            self.world.X, self.world.Y, Z_init,
            cmap='inferno', vmin=0, vmax=1.0, shading='gouraud'
        )
    
        for agent in self.world.agents:
            self.agent_plots[agent.name], = ax.plot(
                [], [], marker='o', color=agent.color,
                markersize=8, markeredgecolor='white', label=agent.name
            )
        ax.legend(loc='lower left')
        self.fig.colorbar(self.heatmap, ax=ax, label='Signal Strength')

    def animate(self, i):
        for agent in self.world.agents:
            if isinstance(agent, Robot):
                agent.update_position(i)
        
        Z_total = self.world.compute_signal_field()
        self.heatmap.set_array(Z_total.ravel())

        
        for agent in self.world.agents:
            self.agent_plots[agent.name].set_data([agent.pos[0]], [agent.pos[1]])

  
        for line in self.links:
            line.remove()
        self.links.clear()

        connected = self.world.connected_robots()
        for idx_a in connected:
            for idx_b in connected:
                if idx_b > idx_a:
                    a = self.world.agents[idx_a]
                    b = self.world.agents[idx_b]
                    line, = self.ax.plot(
                        [a.pos[0], b.pos[0]], [a.pos[1], b.pos[1]],
                        color='cyan', alpha=0.4, linewidth=1.5
                    )
                    self.links.append(line)

        return list(self.agent_plots.values()) + [self.heatmap] + self.links

    def run(self, steps, interval=25, save=False):
        ani = animation.FuncAnimation(
            self.fig, self.animate, frames=steps, 
            interval=interval, blit=False, 
            repeat=False
        )
        if save:
            output_path = "../assets/Multi_signal_and_obstacles.gif"
            ani.save(output_path, writer='pillow', fps=self.fps, dpi=150)
            print("Done")
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help="Enable gif save")
    args = parser.parse_args()


    world = World(resolution=200)
    
    world.add_obstacle(Obstacle((4, 6), 4.0, 1.0, 30.0, 0.1))
    world.add_obstacle(Obstacle((-5, 5), 3.0, 3.0, 0.0, 0.1))

    station = Entity("Station", (0,0), 4.0, 1.0)
    world.add_agent(station)

    frames = 150
    
    path1 = list(zip(np.linspace(9, 7, frames), np.linspace(0, 7, frames)))
    path2 = list(zip(np.linspace(-1, 8, frames), np.linspace(1, 4, frames)))
    path3 = list(zip(np.linspace(-8, -2, frames), np.linspace(8, -2, frames))) 

    world.add_agent(Robot("Robot A", path1, 4.0, 0.7, 'b'))
    world.add_agent(Robot("Robot B", path2, 4.0, 0.7, 'c'))
    world.add_agent(Robot("Robot C", path3, 5.0, 0.7, 'm')) 

    vis = Visualizer(world, fps=25)
    vis.setup()
    
    vis.run(steps=frames, interval=40, save=args.save)