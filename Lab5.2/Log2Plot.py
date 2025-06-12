import re
import matplotlib.pyplot as plt


with open("controller_log_1.txt", "r") as f:
    lines = f.readlines()

# Initialize
timestamps = []
x, y, z, yaw = [], [], [], []
target_x, target_y, target_z, target_yaw = [], [], [], []
cmd_vx, cmd_vy, cmd_vz = [], [], []


pattern = re.compile(
    r"State: \[(.*?)\], Target: \[(.*?)\], dt: (\d+), Vel_cmd: \[(.*?)\]"
)

# Get data from the file
for line in lines:
    match = pattern.search(line)
    if match:
        state_vals = [float(s.split("(")[-1].rstrip(")")) for s in match.group(1).split(", ")]
        x.append(state_vals[0])
        y.append(state_vals[1])
        z.append(state_vals[2])
        yaw.append(state_vals[5])

        target_vals = list(map(float, match.group(2).split(", ")))
        target_x.append(target_vals[0])
        target_y.append(target_vals[1])
        target_z.append(target_vals[2])
        target_yaw.append(target_vals[3])

        timestamps.append(int(match.group(3)))

        vel_cmd_vals = [float(s.split("(")[-1].rstrip(")")) for s in match.group(4).split(", ")]
        cmd_vx.append(vel_cmd_vals[0])
        cmd_vy.append(vel_cmd_vals[1])
        cmd_vz.append(vel_cmd_vals[2])

# Time stamp to relative time
time_s = [(t - timestamps[0]) / 1e3 for t in timestamps]

# Drawing figure
def align_zero(ax1, ax2, data1, data2):

    min1, max1 = min(data1), max(data1)
    min2, max2 = min(data2), max(data2)


    pad1 = (max1 - min1) * 0.1 + 1e-6
    pad2 = (max2 - min2) * 0.1 + 1e-6
    lim1 = [min1 - pad1, max1 + pad1]
    lim2 = [min2 - pad2, max2 + pad2]

    zero_pos1 = -lim1[0] / (lim1[1] - lim1[0])
    zero_pos2 = -lim2[0] / (lim2[1] - lim2[0])

    range2 = lim2[1] - lim2[0]
    shift = (zero_pos2 - zero_pos1) * range2
    ax1.set_ylim(lim1)
    ax2.set_ylim(lim2[0] - shift, lim2[1] - shift)

# Draw double y-axix
def plot_dual_axis(ax, time, actual, target, cmd, label, cmd_label):
    ax.plot(time, actual, label=label)
    ax.plot(time, target, '--', label=f'target_{label}')
    ax.set_ylabel(label)
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(time, cmd, ':r', label=cmd_label, alpha=0.7)
    ax2.set_ylabel(cmd_label + ' (cmd)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')


    align_zero(ax, ax2, actual + target, cmd)


    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')

# Srat drawing
plt.rcParams.update({'font.size': 16})  # Font size

fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

plot_dual_axis(axs[0], time_s, x, target_x, cmd_vx, 'x', 'cmd_vx')
plot_dual_axis(axs[1], time_s, y, target_y, cmd_vy, 'y', 'cmd_vy')
plot_dual_axis(axs[2], time_s, z, target_z, cmd_vz, 'z', 'cmd_vz')

axs[3].plot(time_s, yaw, label='yaw')
axs[3].plot(time_s, target_yaw, '--', label='target_yaw')
axs[3].set_ylabel('yaw')
axs[3].legend()
axs[3].grid(True)
axs[3].set_xlim(min(time_s)-1, max(time_s) + 10)  # Extend x space

axs[3].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()
