import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

matplotlib.use('Qt5Agg')


def plot_colortable(colors, *, ncols=4, sort_colors=True):
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin / width, margin / height,
                        (width - margin) / width, (height - margin) / height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y - 9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig


# plot_colortable(matplotlib.colors.CSS4_COLORS)
# plt.show()


# Read the Excel file
file_path = r"C:\Users\Ruiz\Desktop\New folder\corr_1_20240708141707zernike_coefficients.xlsx"
df = pd.read_excel(file_path)

# Extract the data for plotting
labels = df['mods']
values = df['amps']
str_labels = labels.astype(str)

# Plot the bar graph
plt.ion()
plt.figure(figsize=(5, 3))  # High resolution
plt.bar(str_labels, values, color='deepskyblue')
# Display the plot
plt.xlabel('Zernike Modes', fontsize=12)
plt.ylabel('Amplitudes', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.4)
plt.xticks(rotation=45, ha='center', fontsize=10)
# Tight layout for better spacing
plt.tight_layout()
# Save the figure
plt.savefig(r'C:\Users\Ruiz\Desktop\New folder\zernike_coefficients_c_1.png', dpi=600)  # High-resolution output
plt.show()

# Read the Excel file
file_path = r"C:\Users\Ruiz\Desktop\New folder\corr_2_20240708141822zernike_coefficients.xlsx"
df = pd.read_excel(file_path)

# Extract the data for plotting
labels = df['mods']
values = df['amps']
str_labels = labels.astype(str)

# Plot the bar graph
plt.ion()
plt.figure(figsize=(5, 3))  # High resolution
plt.bar(str_labels, values, color='deepskyblue')
# Display the plot
plt.xlabel('Zernike Modes', fontsize=12)
plt.ylabel('Amplitudes', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.4)
plt.xticks(rotation=45, ha='center', fontsize=10)
# Tight layout for better spacing
plt.tight_layout()
# Save the figure
plt.savefig(r'C:\Users\Ruiz\Desktop\New folder\zernike_coefficients_c_2.png', dpi=600)  # High-resolution output
plt.show()


