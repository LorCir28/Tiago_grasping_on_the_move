import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.interpolate import CubicSpline

# Given data
via_point = (4.5, 1.5)
end_point = (8, 3)
sphere_center = (4, 2)
sphere_radius = 0.05  # 5 cm = 0.05 units
box_top_left = (6.2, 3.7)
box_width = 2.0  # increased length of the box
box_height = 0.85
yaw = 0.56  # radians

# Create waypoints for the trajectory
waypoints_x = [0, 1, 2, 3, 4, 4.2, 4.5, 5, 6, 6.5, 7, 8]
waypoints_y = [0, 0.5, 1, 1.2, 1.4, 1.45, 1.5, 1.8, 2, 2.5, 3, 3]

# Generate smooth trajectory using CubicSpline
cs_x = CubicSpline(np.arange(len(waypoints_x)), waypoints_x)
cs_y = CubicSpline(np.arange(len(waypoints_y)), waypoints_y)

# Create 1000 points for smooth trajectory
num_points = 1000
t = np.linspace(0, len(waypoints_x) - 1, num_points)
smooth_x = cs_x(t)
smooth_y = cs_y(t)

# Function to rotate a point around another point
def rotate_point(x, y, cx, cy, angle):
    s, c = np.sin(angle), np.cos(angle)
    x -= cx
    y -= cy
    x_new = x * c - y * s
    y_new = x * s + y * c
    x_new += cx
    y_new += cy
    return x_new, y_new

# Calculate the corners of the box
box_corners = [
    (box_top_left[0], box_top_left[1]),
    (box_top_left[0] + box_width, box_top_left[1]),
    (box_top_left[0] + box_width, box_top_left[1] - box_height),
    (box_top_left[0], box_top_left[1] - box_height)
]

# Rotate the corners around the top-left corner of the box
box_corners_rotated = [rotate_point(x, y, box_top_left[0], box_top_left[1], yaw) for x, y in box_corners]

# Create the plot
fig, ax = plt.subplots()

# Plot the sphere
sphere = plt.Circle(sphere_center, sphere_radius, color='blue', alpha=0.5, label='Sphere')
ax.add_patch(sphere)

# Plot the rotated box with fill
polygon = plt.Polygon(box_corners_rotated, color='green', alpha=0.5, label='Box')
ax.add_patch(polygon)

# Set the limits and labels
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Robot Trajectory with Sphere and Rotated Box')
ax.legend()

# Initialize a line for the trajectory
line, = ax.plot([], [], marker='o')

# Function to initialize the animation
def init():
    line.set_data([], [])
    return line,

# Function to update the animation at each frame
def update(frame):
    line.set_data(smooth_x[:frame+1], smooth_y[:frame+1])
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_points, init_func=init, blit=True, interval=10, repeat=False)

# Display the animation
plt.show()
